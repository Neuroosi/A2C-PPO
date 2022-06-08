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
BETA = 0.5

def train(states , actions, A, agent, old_agent, optimizer, G):
    pred,values = agent(states)
    old_pred, _ = old_agent(states)
    print(pred==old_pred, values == _)
    values = torch.squeeze(values)
    actions = actions*A.unsqueeze(1)
    pred_ratio = torch.exp(pred- old_pred)
    clip = torch.clamp(pred_ratio, 1-EPSILON, 1+EPSILON)
    entropy = -torch.mean(torch.exp(old_pred)*old_pred)
    loss = BETA*torch.mean((G-values)**2)-torch.mean(torch.min(pred_ratio*actions, clip*actions)) - ALPHA*entropy
    optimizer.zero_grad()
    loss.backward()
    for param in agent.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

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

def predict(agent, state, transition, action_space_size):
    with torch.no_grad():
        state = np.expand_dims(state, axis=0)
        logprob, values = agent(torch.from_numpy(state).float())
        values = torch.squeeze(values).float()
        prob = torch.exp(logprob)
        cache =  torch.squeeze(prob)
        transition.probs.append(cache)
        prob = prob.cpu().detach().numpy()
        prob = np.squeeze(prob)
        return np.random.choice(action_space_size, p = prob), values

def predict_VALUE(agent, state):
    with torch.no_grad():
        return agent(state)

if __name__ == "__main__":
    game = "Breakout-v4"
    env = gym.make(game)
    action_space_size = env.action_space.n
    ##Book keeping
    VALUE_ESTIMATOR_LOSS = []
    POLICY_LOSS = []
    state = deque(maxlen = 4)
    wandb.init(project="PPO_PONG_"+game, entity="neuroori") 
    ##Actors in the simulation
    updater_agent = ACTORCRITIC.NeuralNetwork(action_space_size).to(device)
    actor_agent = ACTORCRITIC.NeuralNetwork(action_space_size).to(device)
    actor_agent.load_state_dict(updater_agent.state_dict())
    ##Optimization stuff
    optimizer = optim.Adam(updater_agent.parameters(), lr = learning_rate)
    ##Transition class
    transition = Transition.Transition(action_space_size)

    ans = input("Use a pretrained model y/n? ")
    if ans == "y":
        loadModel(actor_agent, "AC_WEIGHTS.pth")
        loadModel(updater_agent, "AC_WEIGHTS.pth")
    
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
            action, reward_estimate = predict(actor_agent, makeState(state)/255, transition, action_space_size)
            #if action == 0:
            #    observation, reward, done, info = env.step(2)##UP
            #else:
            #    observation, reward, done, info = env.step(3)##DOWN
            observation, reward, done, info = env.step(action)
            transition.addTransition(makeState(state), reward, action, reward_estimate)
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
        loss_policy = train(states.to(device), actions.to(device),  (G-V_ESTIMATES).to(device), updater_agent, actor_agent, optimizer, G)
        print(loss_policy)
        POLICY_LOSS.append(loss_policy)
        if games_played > 0:
            wandb.log({"BATCH REWARD": batch_reward/games_played})
        else:
            wandb.log({"BATCH REWARD": gamereward})
        games_played = 0
        cumureward = 0
        batch_steps = 0
        transition.resetTransitions()
        actor_agent.load_state_dict(updater_agent.state_dict())
        if total_time % 100000 == 0:
            saveModel(actor_agent, "AC_WEIGHTS.pth")