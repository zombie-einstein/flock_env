import typing
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from flock_env import data_types


class BaseFlockEnv(environment.Environment):
    def __init__(self, reward_func: typing.Callable, n_agents: int):
        self.reward_func = reward_func
        self.n_agents = n_agents

    @property
    def default_params(self) -> data_types.EnvParams:
        return data_types.EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: data_types.EnvState,
        action: Union[int, float],
        params: data_types.EnvParams,
    ) -> Tuple[chex.Array, data_types.EnvState, float, bool, dict]:

        action = jnp.clip(action, a_min=-1.0, a_max=1.0)

        rotations = action[:, 0] * params.max_rotate * jnp.pi
        accelerations = action[:, 1] * params.max_accelerate

        new_headings = (state.agent_headings + rotations) % (2 * jnp.pi)
        new_speeds = jnp.clip(
            state.agent_speeds + accelerations,
            a_min=params.min_speed,
            a_max=params.max_speed,
        )

        d_pos = jax.vmap(lambda a, b: jnp.array([jnp.cos(a) * b, jnp.sin(a) * b]))(
            new_headings, new_speeds
        )
        new_positions = (state.agent_positions + d_pos) % 1.0

        new_state = data_types.EnvState(
            agent_positions=new_positions,
            agent_speeds=new_speeds,
            agent_headings=new_headings,
            time=state.time + 1,
        )

        new_obs = self.get_obs(new_state, params)
        rewards = jax.vmap(self.reward_func, in_axes=(None, 0, None))(
            params, new_positions, new_positions
        )[jnp.newaxis]
        dones = self.is_terminal(state, params)

        return new_obs, new_state, rewards, dones, dict()

    def reset_env(
        self, key: chex.PRNGKey, params: data_types.EnvParams
    ) -> Tuple[chex.Array, data_types.EnvState]:

        k1, k2, k3 = jax.random.split(key, 3)

        agent_positions = jax.random.uniform(k1, (self.n_agents, 2))
        agent_speeds = jax.random.uniform(
            k2, (self.n_agents,), minval=params.min_speed, maxval=params.max_speed
        )
        agent_headings = jax.random.uniform(
            k3, (self.n_agents,), minval=0.0, maxval=2 * jnp.pi
        )

        new_state = data_types.EnvState(
            agent_positions=agent_positions,
            agent_speeds=agent_speeds,
            agent_headings=agent_headings,
            time=0,
        )

        return self.get_obs(new_state, params), new_state

    def get_obs(
        self, state: data_types.EnvState, params: data_types.EnvParams
    ) -> chex.Array:
        raise NotImplementedError

    def is_terminal(
        self, state: data_types.EnvState, params: data_types.EnvParams
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
