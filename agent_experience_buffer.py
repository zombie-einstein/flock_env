from typing import Tuple

import numpy as np
import torch


class AgentReplayMemory:
    def __init__(self, capacity: int, n_agents: int, observation_size: int, device):
        """
        Initialize the agent based experience buffer. This is a
        multi-dimensional buffer containing columns for each agent included in
        the model

        Args:
            capacity (int): Desired capacity of the experience buffer. The
                actual capacity will be rounded up to be divisible by the
                number of agents
            n_agents (int): Number of agents in the model
            observation_size (int): The size of the observation array returned
                by the environment
            device: PyTorch device to push experience samples to
        """
        h = (capacity - capacity % n_agents) // n_agents
        self.capacity = h + 1 if h * n_agents < capacity else h
        self.n_agents = n_agents

        self.states = np.empty(
            (self.capacity, n_agents, observation_size), dtype=np.float32
        )
        self.actions = np.empty((self.capacity, n_agents), dtype=np.int16)[
            :, :, np.newaxis
        ]
        self.rewards = np.empty((self.capacity, n_agents), dtype=np.float32)[
            :, :, np.newaxis
        ]
        self.next_states = np.empty(
            (self.capacity, n_agents, observation_size), dtype=np.float32
        )
        self.dones = np.empty((self.capacity, n_agents), dtype=np.bool)[
            :, :, np.newaxis
        ]

        self.position = 0
        self.length = 0
        self.device = device

    def push_agent_actions(
        self,
        states: np.array,
        actions: np.array,
        rewards: np.array,
        next_states: np.array,
        done: np.array,
    ):
        """
        Insert agent experiences into the buffer. When called new values are
        inserted at the current index, and for each agent. New values are
        inserted on a loop when the end of the buffer is reached

        Args:
            states (np.array): 2d array of previous states for each agent
            actions (np.array): 1d array of actions for each agent
            rewards (np.array): 2d array of reward for each agent
            next_states (np.array): 2d array of next-state observations for
                each agent
            done (np.array): Array of boolean 'done' values for each agent
        """
        self.states[self.position] = states
        self.actions[self.position, :, 0] = actions
        self.rewards[self.position, :, 0] = rewards
        self.next_states[self.position] = next_states
        self.dones[self.position, :, 0] = done

        self.position = (self.position + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)

    def sample(self, batch_size: int) -> Tuple:
        """
        Draw a random sample from the buffer array, drawing uniformly from
        each agents experience

        Args:
            batch_size (int): Number of samples to be drawn

        Returns:
            tuple: Tuple of pytorch tensors containing arrays of
                experience fields
        """
        ix = np.random.randint(0, self.capacity, batch_size)
        iy = np.random.randint(0, self.n_agents, batch_size)

        states = torch.from_numpy(self.states[ix, iy]).float().to(self.device)
        actions = torch.from_numpy(self.actions[ix, iy]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[ix, iy]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[ix, iy]).float().to(self.device)
        dones = (
            torch.from_numpy(self.dones[ix, iy].astype(np.uint8))
            .float()
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def at_capacity(self) -> bool:
        """
        Flag True if the buffer is at capacity, i.e. the columns are full
        Returns:
            bool: True if the buffer has filled at least once
        """
        return self.length == self.capacity

    def __len__(self):
        return self.length
