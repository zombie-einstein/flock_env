from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState:
    agent_positions: jnp.array
    agent_speeds: jnp.array
    agent_headings: jnp.array
    time: int


@struct.dataclass
class EnvParams:
    n_agents: int
    min_speed: float
    max_speed: float
    max_rotate: float
    max_accelerate: float


class FlockEnv(environment.Environment):
    def __init__(self, reward_func):
        self.reward_func = reward_func

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

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

        new_state = EnvState(
            agent_positions=new_positions,
            agent_speeds=new_speeds,
            agent_headings=new_headings,
            time=state.time + 1,
        )

        new_obs = self.get_obs(new_state)

        rewards = jax.vmap(self.reward_func)(new_positions)

        dones = self.is_terminal(state, params)

        return (new_obs, new_state, rewards, dones, dict())

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:

        k1, k2, k3 = jax.random.split(key, 3)

        agent_positions = jax.random.uniform(k1, (params.n_agents, 2))
        agent_speeds = jax.random.uniform(
            k2, (params.n_agents,), minval=params.min_speed, maxval=params.max_speed
        )
        agent_headings = jax.random.uniform(
            k3, (params.n_agents,), minval=0.0, maxval=2 * jnp.pi
        )

        new_state = EnvState(
            agent_positions=agent_positions,
            agent_speeds=agent_speeds,
            agent_headings=agent_headings,
            time=0,
        )

        return self.get_obs(new_state), new_state

    def get_obs(self, state: EnvState) -> chex.Array:
        pass

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        return jnp.full((params.n_boids,), False)

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams):
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams):
        pass

    def state_space(self, params: EnvParams):
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
