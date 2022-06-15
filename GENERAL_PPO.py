import torch
from torch import distributions, nn
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
from wandb import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
##Hyperparameters
learning_rate = 0.0003
GAMMA = 0.99
EPISODES = 5000
BATCH_SIZE = 5
INPUTSIZE = (84,84)
EPSILON = 0.2
ALPHA = 0.01
BETA = 0.5
MAX_ITERS = 4
MAX_UPDATES = 15000
def train(states , actions, A, agent, optimizer, G, old_probs, old_valuess):
    print("LEARNING_RATE ",optimizer.param_groups[0]['lr'])
    indexs = np.arange(len(states))
    total_loss = 0
    total_entropy_loss = 0
    total_policy_loss = 0
    total_values_loss = 0
    for iter in range(MAX_ITERS):
        lower_M = 0
        upper_M = 64
        np.random.shuffle(indexs)
        for m in range(32):
            index = indexs[lower_M:upper_M]
            state = states[index]
            G_ = G[index]
            A_ = A[index]
            actions_ = actions[index]
            pred,values = agent(state)
            new_dist = torch.distributions.Categorical(torch.exp(pred))
            entropies = new_dist.entropy()
            old_pred = old_probs[index]
            old_values = old_valuess[index]
            values = torch.squeeze(values)
            old_pred = torch.squeeze(old_pred)
            actions_ = actions_*A_.unsqueeze(1)
            
            pred_ratio = torch.exp(pred- old_pred)
            clip = torch.clamp(pred_ratio, 1-EPSILON, 1+EPSILON)
            policy_loss = -torch.mean(torch.min(pred_ratio*actions_, clip*actions_))

            clip = old_values + (values - old_values).clamp(-EPSILON, EPSILON)
            values_loss = (G_-values)**2
            clip_loss = (clip-values)**2
            values_loss = torch.max(values_loss, clip_loss)
            values_loss = torch.mean(values_loss)

            entropy_loss = torch.mean(entropies)

            loss = BETA*values_loss+policy_loss - ALPHA*entropy_loss

            optimizer.zero_grad()
            loss.backward()
            for param in agent.parameters():
                param.grad.data.clamp_(-0.5, 0.5)
            optimizer.step()

            lower_M += 64
            upper_M += 64
            total_loss += loss.item()
            total_entropy_loss += entropy_loss
            total_policy_loss += policy_loss
            total_values_loss += values_loss
    return total_loss/(64.0*2), total_entropy_loss/(64.0*2), total_values_loss/(64.0*2), total_policy_loss/(64.0*2)

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

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


    observation = env.reset()
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    gamereward = 0
    games_played = 0
    update = 0
    while update < MAX_UPDATES:
        action, reward_estimate, distribution = predict(actor_agent, makeState(state)/255,  action_space_size)
            #if action == 0:
            #    observation, reward, done, info = env.step(2)##UP
            #else:
            #    observation, reward, done, info = env.step(3)##DOWN
        observation, reward, done, info = env.step(action)
        transition.addTransition(makeState(state), reward, action, reward_estimate, distribution)
        state.append(getFrame(observation))
        total_time += 1
        batch_steps += 1
        gamereward += reward
        env.render()
        if done:
            print("Running reward: ", gamereward)
            wandb.log({"RUNNING REWARD" :  gamereward})
            gamereward = 0
            observation = env.reset()
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            games_played += 1
        if batch_steps % 2048 == 0:
            print("POLICY/VALUES UPDATED", update ,"Gamesplayed: ", games_played, " Steps: ", total_time)
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
            old_rewards = torch.stack(transition.reward_estimate).float()
            total_loss, entropy_loss, values_loss, policy_loss = train(states.to(device), actions.to(device),  (G-V_ESTIMATES).to(device), actor_agent, optimizer, G, old_probs, old_rewards)
            print(total_loss,  values_loss, policy_loss)
            wandb.log({"TOTAL LOSS": total_loss, "ENTROPY LOSS":entropy_loss,"VALUES LOSS": values_loss, "POLICY LOSS":policy_loss})
            batch_steps = 0
            transition.resetTransitions()
            update +=1
            update_linear_schedule(optimizer, update, MAX_UPDATES, learning_rate)
            if total_time % 100000 == 0:
                saveModel(actor_agent, "AC_WEIGHTS.pth")