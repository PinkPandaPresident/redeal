# https://github.com/mahowald/tictactoe/blob/master/tictactoe/model.py

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from collections import namedtuple
from Reinforcement_Learning.creating_data import Game as Game


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """
    Copied verbatim from the PyTorch DQN tutorial.
    During training, observations from the replay memory are
    sampled for policy learning.
    """

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


class Policy(nn.Module):
    """
    Policy model. Consists of a fully connected feedforward
    NN with 3 hidden layers.
    """

    def __init__(self, n_inputs=373, n_outputs=38):
        super(Policy, self).__init__()
        # self.fc1 = nn.Linear(n_inputs, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 2048)
        # self.fc4 = nn.Linear(2048, 1024)
        # self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(512, n_outputs)

        self.fc1 = nn.Linear(n_inputs, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, 1024)
        self.fc9 = nn.Linear(1024, 512)
        self.fc10 = nn.Linear(512, n_outputs)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Forward pass for the model.
        """
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        return x

    def act(self, obf_observation, eps):
        observation = list(obf_observation[0])
        with T.no_grad():

            legal_actions = Game.legal_bids(observation)

            if np.random.random() > eps:
                state = T.tensor([observation]).to(self.device)
                actions = self.forward(state)
                potential_actions = T.tensor(
                    [i if n in legal_actions else -math.inf for n, i in enumerate(list(actions[0]))]
                )

                #
                # new_actions = []
                # for n, i in enumerate(actions[0]):
                #     if n in legal_actions:
                #         new_actions.append(i)
                # new_actions = T.tensor(new_actions)

                action = T.argmax(potential_actions).item()
            else:
                smarter_pool = list(legal_actions) + [35 for i in range(2*len(list(legal_actions)))]

                action = np.random.choice(smarter_pool).item()
            return action




