import numpy as np
import gym


class FlockEnv(gym.Env):
    def __init__(self,
                 n_agents: int,
                 max_v: float,
                 max_dv: float,
                 max_dh: float,
                 n_steps: int,
                 prox_threshold: float = 0.0001):
        """
        Initialize a flock environment

        Args:
            n_agents (int): Number of agents to include in simulation
            max_v (float): Max allowed velocity of agents
            max_dv (float): Max change in velocity
            max_dh (float): Max change in heading
            prox_threshold (float, optional):
        """
        self.n_agents = n_agents

        self.phase_space = np.zeros((4, n_agents))

        self.max_v = max_v
        self.max_dv = max_dv
        self.max_dh = max_dh

        self.n_steps = n_steps
        self.i = 0

        self.prox_threshold = prox_threshold
        self.x_range = np.array([1.0, 1.0])[:, np.newaxis, np.newaxis]

        self.action_space = gym.spaces.box.Box(0, 1.0, shape=(2*self.n_agents,))
        self.observation_space = gym.spaces.box.Box(0, 1.0, shape=(4, self.n_agents))

    def _update_agents(self):
        """update the position of all agents based on current velocity"""
        act_vel = self.max_v*self.phase_space[2]
        angles = (2*self.phase_space[3]-1)*np.pi
        v0 = act_vel * np.cos(angles)
        v1 = act_vel * np.sin(angles)
        self.phase_space[0] = (self.phase_space[0] + v0) % 1
        self.phase_space[1] = (self.phase_space[1] + v1) % 1

    def _rotate_agents(self, rotations):
        """Rotate agent headings from argument rotations"""
        self.phase_space[3] = np.mod(self.phase_space[3] + self.max_dh * rotations, 1)

    def _accelerate_agents(self, accelerations):
        """Update agent speed based on argument accelerations"""
        self.phase_space[2] = np.clip(self.phase_space[2] + accelerations*self.max_dv, 0, 1)

    def _offsets(self):
        """Calculate vector offsets between all agents"""
        x = self.phase_space[:2]
        offs = np.moveaxis(x[np.newaxis, :].T - x, 1, 0)
        return offs[:, ~np.eye(offs.shape[1], dtype=bool)].reshape(2, offs.shape[1], -1)

    def _distance(self, off_arr):
        """Return distances from offset matrix"""
        a = np.abs(off_arr)
        d = np.minimum(a, self.x_range - a)
        return np.sqrt(np.sum(d ** 2, axis=0))

    def reward(self):
        """Calculate reward signals"""
        velocity_score = np.sum(self.phase_space[2])
        distances = self._distance(self._offsets())
        proximity_penalty = - np.sum(distances < self.prox_threshold)
        closeness_bonus = np.sum(np.clip(0.5 - (distances > self.prox_threshold), 0, 1.0))
        return velocity_score + closeness_bonus + proximity_penalty

    def step(self, action):
        self.i += 1
        self._rotate_agents(2*action[:self.n_agents]-1)
        self._accelerate_agents(action[self.n_agents:])
        self._update_agents()
        return self.phase_space, self.reward(), self.i >= self.n_steps, {}

    def reset(self):
        self.phase_space = np.random.random((4, self.n_agents))
        return self.phase_space

    def render(self, mode='human'):
        return self.phase_space[:2].copy()
