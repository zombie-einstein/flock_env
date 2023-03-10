import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from flock_env import base_env, data_types, utils


class SimpleFlockEnv(base_env.BaseFlockEnv):
    def observation_space(self, params: data_types.EnvParams):
        return spaces.Box(-1.0, 1.0, shape=(3,), dtype=jnp.float32)

    def get_obs(
        self, state: data_types.EnvState, params: data_types.EnvParams
    ) -> chex.Array:
        def inner_obs(x, h, x_flock, h_flock):
            vec_2_flock = utils.shortest_vector(x, x_flock, 1.0)
            distances = jnp.sum(jnp.square(vec_2_flock), axis=1)
            range_filter = (distances < params.square_range).astype(jnp.float32)
            n_in_range = jnp.sum(range_filter) - 1.0

            def build_obs():
                dx = range_filter[:, jnp.newaxis] * vec_2_flock
                dx = jnp.sum(dx, axis=0) / n_in_range
                phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
                theta = jnp.sqrt(jnp.sum(jnp.square(dx)) / params.square_range)
                d_phi = utils.shortest_vector(h, phi, 2 * jnp.pi) / jnp.pi
                dh = utils.shortest_vector(h, h_flock, 2 * jnp.pi) / jnp.pi
                dh = jnp.sum(range_filter * dh) / n_in_range
                return jnp.array([theta, d_phi, dh])

            return jax.lax.cond(
                jnp.greater(n_in_range, 0.0), build_obs, lambda: jnp.zeros((3,))
            )

        obs = jax.vmap(inner_obs, in_axes=(0, 0, None, None))(
            state.agent_positions,
            state.agent_headings,
            state.agent_positions,
            state.agent_headings,
        )

        return obs
