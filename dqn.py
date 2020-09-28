import numpy as np

from agent_experience_buffer import AgentReplayMemory
from network import DQN

import torch
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self,
                 state_size,
                 action_size,
                 n_agents,
                 buffer_size: int = 1e5,
                 batch_size: int = 256,
                 gamma: float = 0.995,
                 tau: float = 1e-3,
                 learning_rate: float = 7e-4,
                 update_every: int = 4
                 ):
        """
        Initialize DQN agent using the agent-experience buffer

        Args:
            state_size (int): Size of the state observation returned by the
                environment
            action_size (int): Action space size
            n_agents (int): Number of agents in the environment
            buffer_size (int): Desired total experience buffer size
            batch_size (int): Mini-batch size
            gamma (float): Discount factor
            tau (float): For soft update of target parameters
            learning_rate (float): Learning rate
            update_every (int): Number of steps before target network update
        """

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # Q-Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=learning_rate)
        self.memory = AgentReplayMemory(
            buffer_size, n_agents, state_size, device
        )

        self.t_step = 0

        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def step(self, states, actions, rewards, next_steps, done):

        self.memory.push_agent_actions(
            states, actions, rewards, next_steps, done
        )

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if self.memory.at_capacity():
                experience = self.memory.sample(self.batch_size)
                self.learn(experience, self.gamma)

    def act(self, states, eps=0):
        states = torch.from_numpy(states).float().to(device)
        self.policy_net.eval()

        with torch.no_grad():
            action_values = self.policy_net(states)
        self.policy_net.train()

        r = np.random.random(size=self.n_agents)

        action_values = np.argmax(action_values.cpu().data.numpy(), axis=1)
        random_choices = np.random.randint(
            0, self.action_size, size=self.n_agents
        )

        return np.where(r > eps, action_values, random_choices)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        criterion = torch.nn.MSELoss()
        self.policy_net.train()
        self.target_net.eval()

        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data)
