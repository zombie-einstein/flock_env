from typing import Callable

import chex
import esquilax
import jax.numpy as jnp
from gymnax.environments import spaces

from . import base_env, data_types, steps


class VisionEnv(base_env.BaseFlockEnv):
    """
    Simple flock environment with observations based on the boids model.

    Agents in this version of the environment receive 4 piece of
    information as part of their observations:

    - The angle to the centre of mass of the surrounding flock
    - The distance to the centre of mass of the surrounding flock
    - The vector to the average heading of the surrounding flock
    - The difference from the average speed of the flock

    These reflect the same vectors used in the original boids model to
    steer the individual agents.

    Parameters
    ----------
    reward_func
        Function used to calculate individual agent
        rewards, should have the signature ``f(key, params, x, y)``
        where ``params`` are environment parameters, ``x`` is the
        agent position and ``y`` is positions of another agent.
    n_agents: int
        Number of agents in the environment.
    n_view: int
    """

    def __init__(
        self, reward_func: Callable, n_agents: int, i_range: float, n_view: int = 10
    ):
        self.n_view = n_view
        super().__init__(reward_func, n_agents, i_range)

    def observation_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        Observation space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            Observation space description
        """
        return spaces.Box(0, 1.0, shape=(self.n_view,), dtype=jnp.float32)

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        boids: data_types.Boid,
    ) -> chex.Array:
        """
        Generate agent observations from current state

        Parameters
        ----------
        key: chex.PRNGKey
            JAX random key
        params: flock_env.data_types.EnvParams
            Environment parameters
        boids: flock_env.data_types.EnvState
            Boids States

        Returns
        -------
        chex.Array
            Agent rewards
        """
        obs = esquilax.transforms.spatial(
            steps.view,
            reduction=jnp.minimum,
            default=jnp.ones((self.n_view,)),
            include_self=False,
            n_bins=self.n_bins,
            i_range=self.i_range,
        )(
            key,
            (params.boids.view_angle, params.agent_radius),
            boids,
            boids,
            pos=boids.position,
            n_view=self.n_view,
            i_range=self.i_range,
        )
        return obs
