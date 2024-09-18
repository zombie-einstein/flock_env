import typing

import chex
import esquilax
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from . import data_types, steps


class BaseFlockEnv(
    esquilax.ml.rl.Environment[data_types.EnvState, data_types.EnvParams]
):
    """
    Base flock environment containing core agent update logic

    Agents continuously update their position (wrapped on a torus).
    Actions rotate and accelerate individual agent headings and speeds.
    Agents are initialised to uniformly random initial positions, headings
    and speeds.

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

    def __init__(self, reward_func: typing.Callable, n_agents: int):
        self.reward_func = reward_func
        self.n_agents = n_agents

    def default_params(self) -> data_types.EnvParams:
        """
        Get default environment parameters

        Returns
        -------
        data_types.EnvParams
        """
        return data_types.EnvParams()

    def reset(
        self, key: chex.PRNGKey, params: data_types.EnvParams
    ) -> typing.Tuple[chex.Array, data_types.EnvState]:
        """
        Reset the environment, initialising a new random state

        Parameters
        ----------
        key: chex.PRNGKey
            JAX random key.
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        tuple[chex.Array, flock_env.data_types.EnvState]
            Agent observations and new environment state.
        """
        k1, k2, k3 = jax.random.split(key, 3)

        positions = jax.random.uniform(k1, (self.n_agents, 2))
        speeds = jax.random.uniform(
            k2, (self.n_agents,), minval=params.min_speed, maxval=params.max_speed
        )
        headings = jax.random.uniform(
            k3, (self.n_agents,), minval=0.0, maxval=2.0 * jnp.pi
        )

        new_state = data_types.EnvState(
            boids=data_types.Boid(
                position=positions,
                speed=speeds,
                heading=headings,
            ),
            step=0,
        )
        obs = self.get_obs(key, params, new_state.boids)
        return obs, new_state

    def step(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        state: data_types.EnvState,
        actions: chex.ArrayTree,
    ) -> typing.Tuple[
        chex.ArrayTree, data_types.EnvState, chex.ArrayTree, chex.ArrayTree
    ]:
        """
        Update the state of the environment and calculate rewards

        Update the simulation from argument actions. Broadly the
        steps are:

        - Calculate the new speed and headings of agents from actions
        - Update the positions of the agents
        - Calculate rewards from positions
        - Generate individual agent state observations

        Parameters
        ----------
        key: chex.PRNGKey
            JAX random key.
        params: flock_env.data_types.EnvParams
            Environment parameters
        state: flock_env.data_types.EnvState
            Environment state
        actions: chex.Array
            Array of agent actions

        Returns
        -------
        tuple
            Tuple containing

            - Array of individual agent observations
            - New environment state
            - Agent rewards
            - Terminal flags
        """
        headings, speeds = steps.update_velocity(key, params, (actions, state.boids))
        positions = steps.move(key, params, (state.boids.position, headings, speeds))
        state = data_types.EnvState(
            boids=data_types.Boid(
                position=positions,
                speed=speeds,
                heading=headings,
            ),
            step=state.step + 1,
        )
        rewards = esquilax.transforms.spatial(
            10,
            jnp.add,
            -1.0,
            include_self=False,
            topology="moore",
        )(self.reward_func)(
            key,
            params,
            state.boids.position,
            state.boids.position,
            pos=state.boids.position,
        )
        obs = self.get_obs(key, params, state.boids)
        done = self.is_terminal(params, state)
        return obs, state, rewards, done

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        boids: data_types.Boid,
    ) -> chex.Array:
        """
        Generate agent observations from current state

        Implementations should return a 1d array
        containing values for each individual agent

        Parameters
        ----------
        key: chex.PRNGKey
            JAX random key
        params: flock_env.data_types.EnvParams
            Environment parameters
        boids: flock_env.data_types.Boid
            Agent states.

        Returns
        -------
        chex.Array
            Agent rewards
        """
        raise NotImplementedError

    def is_terminal(
        self, params: data_types.EnvParams, state: data_types.EnvState
    ) -> chex.Array:
        """
        Generate terminal flags from current state

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.
        state: flock_env.data_types.EnvState
            Environment state.

        Returns
        -------
        chex.Array
            Terminal flags.
        """
        return jnp.full((self.n_agents,), False)

    @property
    def num_actions(self) -> int:
        """
        Number of environments actions

        Returns
        -------
        int
        """
        return 2

    def action_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        Action space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            Action space description
        """
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

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
        raise NotImplementedError

    def state_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        State space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            State space description.
        """
        return spaces.Dict(
            dict(
                agent_positions=spaces.Box(0.0, 1.0, (2,), jnp.float32),
                agent_speeds=spaces.Box(
                    params.min_speed,
                    params.max_speed,
                    (),
                    jnp.float32,
                ),
                agent_headings=spaces.Box(
                    0.0,
                    2 * jnp.pi,
                    (),
                    jnp.float32,
                ),
                time=spaces.Discrete(jnp.finfo(jnp.int32).max),
            )
        )
