from typing import Tuple

import numpy as np
import gym
from numba import njit, float32
from numba.types import UniTuple


# Need 32bit versions of π and 2π to keep types consistent
# inside numba functions
TPI = float32(2*np.pi)
PI32 = float32(np.pi)


@njit(float32[:, :](float32[:]))
def _product_difference(a):
    """
    Generates 2d matrix of differences between all pairs in the argument array
    i.e. y[i][j] = x[j]-x[i]  for all i,j where i≠j

    Args:
        a (np.array): 1d array of 32bit floats

    Returns:
        np.array: 2d Array of differences
    """
    n = a.shape[0]
    d = np.empty((n, n-1), dtype=float32)
    for i in range(n):
        for j in range(i):
            d[i][j] = a[j] - a[i]
        for j in range(i+1, n):
            d[i][j-1] = a[j] - a[i]
    return d


@njit(UniTuple(float32[:, :], 2)(float32[:], float32))
def _torus_vectors(a, l):
    """
    Get pairs of vectors between two points, wrapping around the torus. This
    is done for all pairs of points in the argument array

    For example if x=1, y=2 and the torus is length 3, then there are 2 vectors
    from x to y: 1 and -2

    Args:
        a (np.array): 1d array of 32bit floats representing points
        l (float): Length of the torus/closed loop

    Returns:
        tuple(np.array, np.array): Tuple of 2d arrays containing the pairs
            of vectors for all pairs of points from the argument array
    """
    x = _product_difference(a)
    return x, np.sign(x)*(np.abs(x)-l)


@njit(float32[:, :](float32[:], float32))
def _shortest_vec(a, l):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.array): 1d array of 32bit floats representing the points
        l (float): Length of the torus/closed loop

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """
    a, b = _torus_vectors(a, l)
    return np.where(np.abs(a) < np.abs(b), a, b)


@njit(float32[:, :](float32[:, :], float32[:, :]), fastmath=True)
def _distances(xs, ys):
    """Convert x and y vector components to Euclidean distances"""
    return np.sqrt(np.power(xs, 2)+np.power(ys, 2))


@njit(float32[:](float32[:, :], float32), fastmath=True)
def _distance_rewards(d, proximity_threshold):
    """
    Reward function based on distances between agents, for each agent
    the rewards are summed over contributions from all other boids

    Args:
        d (np.array): 2d array of distances between pairs of boids
        proximity_threshold (float): Threshold distance at which boids are
            penalised for being too close

    Returns:
        np.array: 1d array of total rewards for each agent
    """
    distance_rewards = np.exp(-60 * d) - float32(0.001)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if distance_rewards[i][j] < 0:
                distance_rewards[i][j] = 0
            elif d[i][j] < proximity_threshold:
                distance_rewards[i][j] = -100
    return distance_rewards.sum(axis=1)


@njit(float32[:, :](float32[:, :], float32[:, :], float32[:]), fastmath=True)
def _relative_angles(xs, ys, theta):
    """
    Get relative angle between each agents heading and vector to other boids

    Args:
        xs (np.array): 2d array of x-components between pairs of boids
        ys (np.array): 2d array of y-components between pairs of boids:
        theta (np.array): 1d array of agent heading angles

    Returns:
        np.array: 2d array of relative angles between pairs of boids
    """
    angles_to = np.arctan2(ys, xs) + PI32
    a = np.subtract(angles_to, np.expand_dims(theta, -1))
    b = np.sign(a) * (np.abs(a) - TPI)
    return np.where(np.abs(a) < np.abs(b), a, b)/PI32


@njit(float32[:, :](float32[:]), fastmath=True)
def _relative_headings(theta):
    """
    Get smallest angle between heading of all pairs of

    Args:
        theta (np.array): 1d array of 32bit floats representing agent headings
            in radians

    Returns:
        np.array: 2d array of 32bit floats representing relative headings
            for pairs of boids
    """
    return _shortest_vec(theta, TPI) / PI32


class BaseFlockEnv(gym.Env):
    def __init__(self,
                 n_agents: int,
                 max_s: float,
                 n_steps: int,
                 proximity_threshold: float = 0.001):
        """
        Initialize a flock environment

        Args:
            n_agents (int): Number of agents to include in simulation
            max_s (float): Max allowed velocity of agents
            n_steps (int): Number of steps in an episode
            proximity_threshold (float, optional): Distance at which boids are
                penalised for being too close to other boids
        """
        self.n_agents = n_agents

        # These arrays form the phase space of the flock
        # Position co-ords in 2d range [0, 1]
        self.x = np.zeros((2, n_agents), dtype=np.float32)
        # Speed in range [0, 1]
        self.speed = np.zeros(n_agents, dtype=np.float32)
        # Heading in range [0, 2π]
        self.theta = np.zeros(n_agents, dtype=np.float32)

        self.max_s = max_s
        self.n_steps = n_steps
        self.i = 0
        self.proximity_threshold = proximity_threshold
        self.max_distance = np.sqrt(2*(0.5**2))

        # The standard observation space is a local view on the phase space
        # for each agent i.e. 4 phase-space values x each other boid
        self.observation_space = gym.spaces.box.Box(
            -1.0, 1.0, shape=(4*(self.n_agents-1),)
        )

    def _update_agents(self):
        """
        Update the position of all agents based on current
        speed and headings
        """
        act_vel = self.max_s*self.speed
        angles = self.theta-np.pi
        v0 = act_vel * np.cos(angles)
        v1 = act_vel * np.sin(angles)
        self.x[0] = (self.x[0] + v0) % 1
        self.x[1] = (self.x[1] + v1) % 1

    def _accelerate_agents(self, accelerations: np.array):
        """
        Should implement a function that accelerates/decelerates each (and
        every) agents based on argument actions

        Args:
            accelerations (np.array): Array of actions for each agent
        """
        raise NotImplementedError

    def _rotate_agents(self, rotations: np.array):
        """
        Should implement a function that rotates (i.e. steers) the agents
        based on argument actions
        Args:
            rotations (np.array): Array of actions for each agent
        """
        raise NotImplementedError

    def _observe(self) -> np.array:
        """
        Should return array of local observations for each agent

        Returns:
            np.array: Array of observations
        """
        raise NotImplementedError

    def _rewards(self, ds: np.array) -> np.array:
        """
        Should return array of rewards for each agent

        Args:
            ds (np.array): 2d array of distance between pairs of agents

        Returns:
            np.array: Array  of rewards for each agent
        """
        raise NotImplementedError

    def step(self, actions: np.array) -> Tuple:
        """
        As per over open AI API this should advance the model one step and
        return a tuple containing the updated observation, rewards and done
        flag

        Args:
            actions (np.array): Array of actions, containing actions to apply
            to each agent in the flock

        Returns:
            tuple: Tuple in the format (local_observations, rewards, done, {})
                as per the open AI API
        """
        raise NotImplementedError

    def reset(self) -> np.array:
        """
        As per the open AI API, reset the state of the agents and return the
        observation of the new state

        In this case the base default resets all the phase space variables to


        Returns:
            array of local observations for each agent
        """
        self.x = np.random.random(size=(2, self.n_agents))
        self.speed = np.random.random(self.n_agents)
        self.theta = TPI*np.random.random(self.n_agents)
        self.i = 0

        _, _, local_observations = self._observe()

        return local_observations

    def render(self, mode='human'):
        pass


class DiscreteActionFlock(BaseFlockEnv):
    def __init__(self,
                 n_agents: int,
                 speed: float,
                 n_steps: int,
                 rotation_size: float,
                 n_actions: int,
                 proximity_threshold: float = 0.001,
                 n_nearest: int = 10):
        """
        Initialize a discrete action flock environment

        In this environment the boids are only allowed to rotate by a fixed
        amount at each step, the action space is then discrete values indexing
        these rotations

        Args:
            n_agents (int): Number of agents to include in simulation
            speed (float): Max allowed velocity of agents
            n_steps (int): Number of steps in an episode
            rotation_size (float): Smallest rotation size in radians
            n_actions (int): NUmber of allowed rotations actions, should be an
                odd integer >1
            proximity_threshold (float, optional): Distance at which other
                boids are considered too close for reward
            n_nearest (int): Number of agents to include in the local
                observations generated for each agent
        """
        assert n_actions % 2 == 1, f"Number of actions must be an odd integer got {n_actions}"
        assert n_nearest <= n_agents, f"Number of agents included in observation should be <= number of agents"

        super(DiscreteActionFlock, self).__init__(
            n_agents,
            speed,
            n_steps,
            proximity_threshold=proximity_threshold)

        mid = (n_actions-1)//2

        self.n_actions = n_actions
        self.n_nearest = n_nearest
        self.rotations = PI32*np.arange(-mid, mid+1).astype(np.float32)*rotation_size
        self.observation_space = gym.spaces.box.Box(-1.0, 1.0, shape=(3 * n_nearest,))
        self.action_space = gym.spaces.Discrete(self.rotations.shape[0])

    def _rotate_agents(self, actions: np.array):
        """
        Rotate the agents according to the argument actions indices
        Args:
            actions (np.array): Array of actions indexing the amount to
                rotate (steer) each of the agents by
        """
        self.theta = np.mod(self.theta + self.rotations[actions], TPI)

    def _accelerate_agents(self, actions: np.array):
        """Agents move at a fixed speed so should not be used"""
        raise NotImplementedError

    def _rewards(self, d: np.array) -> np.array:
        """
        Get rewards for each agent based on distances to other boids

        Args:
            d (np.array): 2d array representing euclidean distances between
                each pair of boids

        Returns:
            np.array: 1d array of reward values for each agent
        """
        return _distance_rewards(d, self.proximity_threshold)

    def _observe(self) -> np.array:
        """
        Returns a view on the flock phase space local to each agent. Since
        in this case all the agents move at the same speed we return the
        x and y components of vectors relative to each boid and the relative
        heading relative to each agent.

        In order for the agents to have similar observed states, for each agent
        neighbouring boids are sorted in distance order and then the closest
        neighbours included in the observation space

        Returns:
            np.array: Array of local observations for each agent, bounded to
                the range [-1,1]
        """
        xs = _shortest_vec(self.x[0], 1.0)
        ys = _shortest_vec(self.x[1], 1.0)
        d = _distances(xs, ys)

        # Sorted indices of flock members by distance
        sort_idx = np.argsort(d, axis=1)[:, :self.n_nearest]

        relative_headings = _relative_headings(self.theta)

        closest_x = np.take_along_axis(xs, sort_idx, axis=1)
        closest_y = np.take_along_axis(ys, sort_idx, axis=1)
        closest_h = np.take_along_axis(relative_headings, sort_idx, axis=1)

        local_observation = np.concatenate([2*closest_x,
                                            2*closest_y,
                                            closest_h], axis=1)

        return d, local_observation

    def step(self, actions: np.array) -> Tuple:
        """
        Step the model forward updating applying the steering actions to the
        agents, then updating the positions of the boids

        Args:
            actions (np.array): Array of steering actions applied to each agent
                actions index the array of discrete values

        Returns:
            tuple: Tuple in the format (local_observations, rewards, done, {})
                as per the open AI API
        """
        self._rotate_agents(actions)
        self._update_agents()
        self.i += 1

        d, local_observations = self._observe()
        rewards = self._rewards(d)

        return local_observations, rewards, self.i >= self.n_steps, {}

    def reset(self) -> np.array:
        """
        Reset the environment assigning the agents random positions and
        headings but assigning them all the max allowed speed

        Returns:
            np.array: Array of local observations of the reset state
        """
        self.x = np.random.random(size=(2, self.n_agents)).astype(np.float32)
        self.speed = np.ones(self.n_agents).astype(np.float32)
        self.theta = TPI * np.random.random(self.n_agents).astype(np.float32)
        self.i = 0

        _, local_observations = self._observe()

        return local_observations
