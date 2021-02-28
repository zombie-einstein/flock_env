from typing import List, Tuple

import gym
import numpy as np
from numba import float32, int32, njit

# Need 32bit versions of π and 2π to keep types consistent
# inside numba functions
TPI = float32(2 * np.pi)
PI32 = float32(np.pi)


@njit(float32[:, :](float32[:], int32), nogil=True)
def _product_difference(a, n):
    """
    Generates 2d matrix of differences between all pairs in the argument array
    i.e. y[i][j] = x[j]-x[i]  for all i,j where i≠j

    Args:
        a (np.array): 1d array of 32bit floats
        n (int): Number of entries in array that will form 1st index of result

    Returns:
        np.array: 2d Array of differences
    """
    m = a.shape[0]
    d = np.empty((n, m - 1), dtype=float32)
    for i in range(n):
        for j in range(i):
            d[i][j] = a[j] - a[i]
        for j in range(i + 1, m):
            d[i][j - 1] = a[j] - a[i]
    return d


@njit(float32[:, :](float32[:], float32, int32), nogil=True)
def _shortest_vecs(a, length, m):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.array): 1d array of 32bit floats representing the points
        length (float): Length of the torus/closed loop
        m: (int): The width of the array

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """

    x = _product_difference(a, m)
    x_ = np.sign(x) * (np.abs(x) - length)
    return np.where(np.abs(x) < np.abs(x_), x, x_)


@njit(float32(float32, float32, float32), nogil=True)
def _shortest_vec(a, b, length):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.float32):
        b (np.float32):

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """
    x = a - b
    x_ = np.sign(x) * (np.abs(x) - length)
    return x if np.abs(x) < np.abs(x_) else x_


@njit(float32[:, :](float32[:, :], float32[:, :]), fastmath=True, nogil=True)
def _distances(xs, ys):
    """Convert x and y vector components to Euclidean distances"""
    return np.sqrt(np.power(xs, 2) + np.power(ys, 2))


@njit(float32[:](float32[:, :], float32, float32), fastmath=True, nogil=True)
def _distance_rewards(d, proximity_threshold, distant_threshold):
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
    distance_rewards = np.zeros(d.shape, dtype=float32)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if proximity_threshold > d[i][j] < distant_threshold:
                distance_rewards[i][j] = np.exp(-40 * d[i][j])
    return distance_rewards.sum(axis=1)


@njit(
    float32[:, :](float32[:, :], float32[:, :], float32[:]), fastmath=True, nogil=True
)
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
    return np.where(np.abs(a) < np.abs(b), a, b) / PI32


@njit(float32[:, :](float32[:]), fastmath=True, nogil=True)
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
    return _shortest_vecs(theta, TPI, theta.shape[0]) / PI32


@njit(float32(float32, float32), fastmath=True, nogil=True)
def _relative_heading(a, b):
    """
    Get smallest angle between heading of all pairs of

    Args:
        a (np.float32):
        b (np.float32):

    Returns:
        np.array: 2d array of 32bit floats representing relative headings
            for pairs of boids
    """
    return _shortest_vec(a, b, TPI) / PI32


@njit(
    (float32[:, :], float32[:], int32, int32, int32, float32), fastmath=True, nogil=True
)
def _observe(pos, theta, n_agents, n_obstacles, n_nearest, max_distance):
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
    xs = _shortest_vecs(pos[0], 1.0, n_agents)
    ys = _shortest_vecs(pos[1], 1.0, n_agents)
    d = _distances(xs, ys)

    obs_width = 3 * n_nearest + 2 * n_obstacles

    local_observation = np.zeros((n_agents, obs_width), dtype=float32)
    # relative_headings = _relative_headings(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    obs_x = xs[:, n_agents - 1 :]
    obs_y = ys[:, n_agents - 1 :]

    for i in range(n_agents):
        sort_idx = np.argsort(d[i, : n_agents - 1])[:n_nearest]
        cos_ti = cos_t[i]
        sin_ti = sin_t[i]
        xs_i = xs[i]
        ys_i = ys[i]
        obs_x_i = obs_x[i]
        obs_y_i = obs_y[i]
        theta_i = theta[i]
        for j in range(n_nearest):
            n = sort_idx[j]
            local_observation[i, n] = (
                cos_ti * xs_i[n] + sin_ti * ys_i[n]
            ) / max_distance
            local_observation[i, n_nearest + j] = (
                cos_ti * ys_i[n] - sin_ti * xs_i[n]
            ) / max_distance
            local_observation[i, 2 * n_nearest + j] = _relative_heading(
                theta_i, theta[n]
            )

        for k in range(n_obstacles):
            obs_x_ = (cos_ti * obs_x_i[k] + sin_ti * obs_y_i[k]) / max_distance
            obs_y_ = (cos_ti * obs_y_i[k] - sin_ti * obs_x_i[k]) / max_distance
            local_observation[i, 3 * n_nearest + k] = obs_x_
            local_observation[i, 3 * n_nearest + n_obstacles + k] = obs_y_

    return d, local_observation


class BaseFlockEnv(gym.Env):
    def __init__(
        self,
        n_agents: int,
        max_s: float,
        n_steps: int,
        flock_reward_scaling: float,
        obstacle_penalty_scaling: float,
        proximity_threshold: float = 0.001,
        obstacles: List[Tuple] = (),
    ):
        """
        Initialize a flock environment

        Args:
            n_agents (int): Number of agents to include in simulation
            max_s (float): Max allowed velocity of agents
            n_steps (int): Number of steps in an episode
            proximity_threshold (float, optional): Distance at which boids are
                penalised for being too close to other boids
            obstacles (list[tuple]): Tuples of triples containing the centre of
                the obstacles (x and y) and radius of environmental obstacles
        """
        self.n_agents = n_agents
        self.n_obstacles = len(obstacles)

        # These arrays form the phase space of the flock
        # Position co-ords in 2d range [0, 1]
        self.x = np.zeros((2, n_agents + self.n_obstacles), dtype=np.float32)
        self.x[:, self.n_agents :] = np.array([i[:2] for i in obstacles]).T.astype(
            np.float32
        )
        # Speed in range [0, 1]
        self.speed = np.zeros(n_agents, dtype=np.float32)
        # Heading in range [0, 2π]
        self.theta = np.zeros(n_agents, dtype=np.float32)

        self.obstacle_radii = np.array([i[2] for i in obstacles])[np.newaxis, :].astype(
            np.float32
        )

        self.max_s = max_s
        self.n_steps = n_steps
        self.flock_reward_scaling = flock_reward_scaling
        self.obstacle_penalty_scaling = obstacle_penalty_scaling
        self.i = 0
        self.proximity_threshold = proximity_threshold
        self.max_distance = np.sqrt(2 * (0.5 ** 2))

        # The standard observation space is a local view on the phase space
        # for each agent i.e. 4 phase-space values x each other boid
        self.observation_space = gym.spaces.box.Box(
            -1.0, 1.0, shape=(4 * (self.n_agents - 1) + 2 * self.n_obstacles,)
        )

    def _update_agents(self):
        """
        Update the position of all agents based on current
        speed and headings
        """
        act_vel = self.max_s * self.speed
        v0 = act_vel * np.cos(self.theta)
        v1 = act_vel * np.sin(self.theta)
        self.x[0][: self.n_agents] = (self.x[0][: self.n_agents] + v0) % 1
        self.x[1][: self.n_agents] = (self.x[1][: self.n_agents] + v1) % 1

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

    def _obstacle_penalties(self, ds: np.array):
        """
        Return penalties for agent colliding with obstacles

        Args:
            ds (np.array): 2d array distances to obstacles for each agent

        Returns:

        """
        return np.any(ds < self.obstacle_radii, axis=1)

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
        self.theta = TPI * np.random.random(self.n_agents)
        self.i = 0

        _, _, local_observations = self._observe()

        return local_observations

    def render(self, mode="human"):
        pass


class DiscreteActionFlock(BaseFlockEnv):
    def __init__(
        self,
        n_agents: int,
        speed: float,
        n_steps: int,
        flock_reward_scaling: float,
        obstacle_penalty_scaling: float,
        rotation_size: float,
        n_actions: int,
        distant_threshold: float = 0.01,
        proximity_threshold: float = 0.001,
        n_nearest: int = 10,
        obstacles: List[Tuple] = (),
    ):
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
            distant_threshold (float): Distance cut-off for rewards
            proximity_threshold (float, optional): Distance at which other
                boids are considered too close for reward
            n_nearest (int): Number of agents to include in the local
                observations generated for each agent
        """
        assert (
            n_actions % 2 == 1
        ), f"Number of actions must be an odd integer got {n_actions}"
        assert (
            n_nearest <= n_agents
        ), "Number of agents in observation should be <= number of agents"
        assert distant_threshold > proximity_threshold

        super(DiscreteActionFlock, self).__init__(
            n_agents,
            speed,
            n_steps,
            flock_reward_scaling=flock_reward_scaling,
            obstacle_penalty_scaling=obstacle_penalty_scaling,
            proximity_threshold=proximity_threshold,
            obstacles=obstacles,
        )

        mid = (n_actions - 1) // 2

        self.proximity_threshold = float32(self.proximity_threshold)
        self.distant_threshold = float32(distant_threshold)
        self.n_actions = n_actions
        self.n_nearest = n_nearest
        self.rotations = (
            PI32 * np.arange(-mid, mid + 1).astype(np.float32) * rotation_size
        )

        observation_shape = (3 * n_nearest) + (2 * self.n_obstacles)

        self.observation_space = gym.spaces.box.Box(
            -1.0, 1.0, shape=(observation_shape,)
        )
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
        agent_rewards = self.flock_reward_scaling * _distance_rewards(
            d[:, : self.n_agents - 1], self.proximity_threshold, self.distant_threshold,
        )
        obstacle_penalties = self.obstacle_penalty_scaling * self._obstacle_penalties(
            d[:, self.n_agents - 1 :]
        )
        return agent_rewards - obstacle_penalties

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
        return _observe(
            self.x,
            self.theta,
            self.n_agents,
            self.n_obstacles,
            self.n_nearest,
            self.max_distance,
        )

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
        self.x[:, : self.n_agents] = np.random.random(size=(2, self.n_agents)).astype(
            np.float32
        )
        self.speed = np.ones(self.n_agents).astype(np.float32)
        self.theta = TPI * np.random.random(self.n_agents).astype(np.float32)
        self.i = 0

        _, local_observations = self._observe()

        return local_observations
