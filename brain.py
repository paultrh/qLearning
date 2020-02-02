from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import torch
from torch import nn
from torch import optim
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from test import Game

print("PyTorch:\t{}".format(torch.__version__))

# input score
# input sensors status

# [N, S, E, W, sNW, sSE, sSW, sNE, bNW, bSE, bSW, bNE]
ex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sensors_nb = 12
action_nb = 4
# output = [HAUT, BAS, DROITE, GAUCHE]


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.n_inputs = 12
        self.n_outputs = 4

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs))

    def forward(self, state):
        print("state", state)
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0

source = MyNN()
target = MyNN()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            _, max_index = torch.max(source(state), 0)
            print(source(state))
            print('max_index', max_index)
            l = []
            for i in range(0, 4):
                if i == max_index:
                    l.append(1)
                else:
                    l.append(0)
            return torch.tensor(l, dtype=torch.long)
    else:
        r = random.randrange(4)
        l = []
        print('r', r)
        for i in range(0, 4):
            if i == r:
                l.append(1)
            else:
                l.append(0)
        return torch.tensor(l, dtype=torch.long)


episode_durations = []
optimizer = optim.RMSprop(source.parameters())
memory = ReplayMemory(10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    print('state_batch', state_batch)
    print('action_batch', action_batch)
    print('reward_batch', reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = source(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    #next_state_values = torch.zeros(BATCH_SIZE)
    #next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    #expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in source.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


GAME = Game()
num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state

    last_screen = torch.FloatTensor(GAME.get_initial_state())
    current_screen = torch.FloatTensor(GAME.get_state())
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        print(action)
        _, reward, done, _ = GAME.step(action.tolist().index(1))
        reward = torch.tensor([reward], dtype=torch.long)

        # Observe new state
        last_screen = current_screen
        current_screen = torch.FloatTensor(GAME.get_state())
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        print(state)
        print(action)
        print(next_state)
        print(reward)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target.load_state_dict(source.state_dict())


print('Complete')


'''
env = gym.make('CartPole-v0')
policy_est = MyNN(env)
rewards = reinforce(env, policy_est)
'''