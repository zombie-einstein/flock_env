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

    Parameters
    ----------
    reward_func
        Function used to calculate individual agent
        rewards, should have the signature ``f(key, params, x, y)``
        where ``params`` are environment parameters, ``x`` is the
        agent position and ``y`` is positions of another agent.
    n_agents: int
        Number of agents in the environment.
    """

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
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=jnp.float32)

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        state: data_types.EnvState,
    ) -> chex.Array:
        """
        Generate agent observations from current state

        Observes the average location of the local
        flock, and relative heading and speed.
        Then converts these into an observation
        in polar co-ordinates.

        Parameters
        ----------
        key: chex.PRNGKey
            JAX random key
        params: flock_env.data_types.EnvParams
            Environment parameters
        state: flock_env.data_types.EnvState
            Environment state

        Returns
        -------
        chex.Array
            Agent rewards
        """
        n_nb, x_nb, s_nb, h_nb = steps.observe(
            key, params, state, state, pos=state.agent_positions
        )
        obs = steps.flatten_observations(key, params, (state, n_nb, x_nb, s_nb, h_nb))
        return obs
