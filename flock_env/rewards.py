import chex
import esquilax
import jax.lax
import jax.numpy as jnp

from flock_env import data_types


def exponential_rewards(
    _k: chex.PRNGKey, params: data_types.EnvParams, x: chex.Array, y: chex.Array
) -> chex.Array:

    d = esquilax.utils.shortest_distance(x, y, 1.0, norm=True)

    reward = jax.lax.cond(
        d < params.agent_radius,
        lambda _d: -params.collision_penalty,
        lambda _d: jnp.exp(-50 * _d),
        d,
    )

    return reward
