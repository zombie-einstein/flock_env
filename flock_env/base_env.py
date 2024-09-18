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

    Args:
        reward_func: Function used to calculate individual agent
            rewards, should have the signature ``f(params, x, x_flock)``
            where ``params`` are environment parameters, ``x`` is the
            agent position and ``x_flock`` is positions of the whole flock.
        n_agents (int): Number of agents in the environment.
    """

    def __init__(self, reward_func: typing.Callable, n_agents: int):
        self.reward_func = reward_func
        self.n_agents = n_agents

    def default_params(self) -> data_types.EnvParams:
        return data_types.EnvParams()

    def reset(
        self, key: chex.PRNGKey, params: data_types.EnvParams
    ) -> typing.Tuple[chex.Array, data_types.EnvState]:
        k1, k2, k3 = jax.random.split(key, 3)

        agent_positions = jax.random.uniform(k1, (self.n_agents, 2))
        agent_speeds = jax.random.uniform(
            k2, (self.n_agents,), minval=params.min_speed, maxval=params.max_speed
        )
        agent_headings = jax.random.uniform(
            k3, (self.n_agents,), minval=0.0, maxval=2.0 * jnp.pi
        )

        new_state = data_types.EnvState(
            agent_positions=agent_positions,
            agent_speeds=agent_speeds,
            agent_headings=agent_headings,
            time=jnp.zeros((self.n_agents,), dtype=jnp.int32),
        )
        obs = self.get_obs(key, params, new_state)
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
        headings, speeds = steps.update_velocity(key, params, (actions, state))
        positions = steps.move(key, params, (state.agent_positions, headings, speeds))
        state = data_types.EnvState(
            agent_positions=positions,
            agent_speeds=speeds,
            agent_headings=headings,
            time=state.time + 1,
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
            state.agent_positions,
            state.agent_positions,
            pos=state.agent_positions,
        )
        obs = self.get_obs(key, params, state)
        done = self.is_terminal(params, state)
        return obs, state, rewards, done

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.EnvParams,
        state: data_types.EnvState,
    ) -> chex.Array:
        raise NotImplementedError

    def is_terminal(
        self, params: data_types.EnvParams, state: data_types.EnvState
    ) -> chex.Array:
        return jnp.full((self.n_agents,), False)

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: data_types.EnvParams):
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: data_types.EnvParams):
        raise NotImplementedError

    def state_space(self, params: data_types.EnvParams):
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
