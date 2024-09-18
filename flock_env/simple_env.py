import chex
import jax.numpy as jnp
from gymnax.environments import spaces

from . import base_env, data_types, steps


class SimpleFlockEnv(base_env.BaseFlockEnv):
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

    Attributes:
        reward_func: Function used to calculate individual agent
            rewards, should have the signature ``f(params, x, x_flock)``
            where ``params`` are environment parameters, ``x`` is the
            agent position and ``x_flock`` is positions of the whole flock.
        n_agents (int): Number of agents in the environment.
    """

    def observation_space(self, params: data_types.EnvParams):
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=jnp.float32)

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        state: data_types.EnvState,
    ) -> chex.Array:
        n_nb, x_nb, s_nb, h_nb = steps.observe(
            key, params, state, state, pos=state.agent_positions
        )
        obs = steps.flatten_observations(key, params, (state, n_nb, x_nb, s_nb, h_nb))
        return obs
