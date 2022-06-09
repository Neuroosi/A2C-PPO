import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import gym
import math
from collections import deque
import skimage
import ACTORCRITIC
import Transition
import copy
from wandb import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
##Hyperparameters
learning_rate = 0.00025
GAMMA = 0.99
EPISODES = 5000
BATCH_SIZE = 5
INPUTSIZE = (84,84)
EPSILON = 0.2
ALPHA = 0.01
BETA = 1
MAX_ITERS = 5
KL_LIMES = 0.01
def train(states , actions, A, agent,  optimizer, G, entropies, old_pred):
    indexs = np.arange(len(states))
    for iter in range(MAX_ITERS):
        estimate_kl_div = 0
        lower_M = 0
        upper_M = 1000
        np.random.shuffle(indexs)
        for step in range(5):
            index = indexs[lower_M:upper_M]
            state = states[index]
            G_ = G[index]
            A_ = A[index]
            actions_= actions[index]
            pred,values = agent(state)
            old_pred_ = old_pred[index]
            values = torch.squeeze(values)
            actions_ = actions_*A_.unsqueeze(1)
            pred_ratio = torch.exp(pred- old_pred_)
            clip = torch.clamp(pred_ratio, 1-EPSILON, 1+EPSILON)
            entropy_loss = -torch.mean(entropies)
            values_loss = torch.mean((G_-values)**2)
            policy_loss = -torch.mean(torch.min(pred_ratio*actions_, clip*actions_))
            loss = BETA*values_loss+policy_loss + ALPHA*entropy_loss
            optimizer.zero_grad()
            loss.backward()
            for param in agent.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            lower_M += 1000
            upper_M += 1000
            estimate_kl_div += torch.mean(torch.exp(pred)*(pred-old_pred_))
        #estimate_kl_div /= 20
        if estimate_kl_div > KL_LIMES:
            print("breaking at iter ",  iter)
            break
    return loss.item(), entropy_loss, values_loss, policy_loss

def getFrame(x):
    x = x[35:210,0:160]
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, INPUTSIZE)
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state = state.astype('uint8')
    return state

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)

def saveModel(agent, filename):
    torch.save(agent.state_dict(), filename)
    print("Model saved!")

def loadModel(agent, filename):
    agent.load_state_dict(torch.load(filename))
    print("Model loaded!")

def predict(agent, state, action_space_size):
    with torch.no_grad():
        state = np.expand_dims(state, axis=0)
        logprob, values = agent(torch.from_numpy(state).float())
        values = torch.squeeze(values).float()
        prob = torch.exp(logprob)
        prob = prob.cpu().detach().numpy()
        prob = np.squeeze(prob)
        return np.random.choice(action_space_size, p = prob), values, logprob

def predict_VALUE(agent, state):
    with torch.no_grad():
        return agent(state)

if __name__ == "__main__":
    game = "BreakoutDeterministic-v4"
    env = gym.make(game)
    action_space_size = env.action_space.n
    ##Book keeping
    VALUE_ESTIMATOR_LOSS = []
    POLICY_LOSS = []
    state = deque(maxlen = 4)
    wandb.init(project="PPO_PONG_"+game, entity="neuroori") 
    ##Actors in the simulation
    actor_agent = ACTORCRITIC.NeuralNetwork(action_space_size).to(device)

    ##Optimization stuff
    optimizer = optim.Adam(actor_agent.parameters(), lr = learning_rate)
    ##Transition class
    transition = Transition.Transition(action_space_size)

    ans = input("Use a pretrained model y/n? ")
    if ans == "y":
        loadModel(actor_agent, "AC_WEIGHTS.pth")
    
    total_time = 0
    batch_steps = 0

    for episode in range(1,EPISODES+500000000000):
        observation = env.reset()
        state.append(getFrame(observation))
        state.append(getFrame(observation))
        state.append(getFrame(observation))
        state.append(getFrame(observation))
        gamereward = 0
        games_played = 0
        batch_reward = 0
        while batch_steps < 5000:
            action, reward_estimate, distribution = predict(actor_agent, makeState(state)/255, action_space_size)
            #if action == 0:
            #    observation, reward, done, info = env.step(2)##UP
            #else:
            #    observation, reward, done, info = env.step(3)##DOWN
            observation, reward, done, info = env.step(action)
            entropy = -torch.sum(distribution*torch.exp(distribution))
            transition.addTransition(makeState(state), reward, action, reward_estimate, entropy, distribution)
            state.append(getFrame(observation))
            total_time += 1
            batch_steps += 1
            gamereward += reward
            env.render()
            if done:
                print("Running reward: ", gamereward)
                batch_reward += gamereward
                gamereward = 0
                observation = env.reset()
                state.append(getFrame(observation))
                state.append(getFrame(observation))
                state.append(getFrame(observation))
                state.append(getFrame(observation))
                games_played += 1

        if games_played > 0:
            print("Batch running reward: ", batch_reward/games_played, " Episode: ", episode, " Steps: ", total_time)
        else:
            print("Batch running reward: ", gamereward, " Episode: ", episode, " Steps: ", total_time)
        ##Put data to a tensor form
        G = transition.discounted_reward(GAMMA)
        G = torch.from_numpy(G).to(device).float()
        states = [torch.from_numpy(np.array(state)/255) for state in transition.states]
        states = torch.stack(states)
        states = states.float()
        actions = [torch.from_numpy(np.array(action)) for action in transition.actions]
        actions = torch.stack(actions)
        actions = actions.float()
        ##TRAIN
        V_ESTIMATES = torch.stack(transition.reward_estimate)
        V_ESTIMATES = V_ESTIMATES.float()
        old_probs = torch.stack(transition.old_probs).float()
        total_loss, entropy_loss, values_loss, policy_loss = train(states.to(device), actions.to(device),  (G-V_ESTIMATES).to(device),  actor_agent, optimizer, G, torch.stack(transition.entropies).float(), old_probs)
        print(total_loss, entropy, values_loss, policy_loss)
        if games_played > 0:
            wandb.log({"BATCH REWARD": batch_reward/games_played, "TOTAL LOSS": total_loss, "ENTROPY LOSS":entropy_loss,"VALUES LOSS": values_loss, "POLICY LOSS":policy_loss})
        else:
            wandb.log({"BATCH REWARD": batch_reward/games_played, "TOTAL LOSS": total_loss, "ENTROPY LOSS":entropy_loss,"VALUES LOSS": values_loss, "POLICY LOSS":policy_loss})
        games_played = 0
        cumureward = 0
        batch_steps = 0
        transition.resetTransitions()
        if total_time % 100000 == 0:
            saveModel(actor_agent, "AC_WEIGHTS.pth")