import numpy as np

class Transition:
    def __init__(self, actionspacesize):
        self.actionSpaceSize = actionspacesize
        self.states = []
        self.rewards = []
        self.actions = []
        self.old_probs = []
        self.reward_estimate = []

    def addTransition(self, state, reward, action, reward_estimate):
        self.states.append(state)
        self.rewards.append(reward)
        cache = np.zeros(self.actionSpaceSize)
        cache[action] = 1
        self.actions.append(cache)
        self.reward_estimate.append(reward_estimate)


    def resetTransitions(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.old_probs = []
        self.reward_estimate = []
        
    def discounted_reward(self, GAMMA):
        G = np.zeros(len(self.rewards))
        ##Calculate discounted reward
        cache = 0
        for t in reversed(range(0, len(self.rewards))):
            if self.rewards[t] != 0: cache = 0
            cache = cache*GAMMA + self.rewards[t]
            G[t] = cache
        ##Normalize
        G = (G-np.mean(G))/(np.std(G)+1e-8)
        return G