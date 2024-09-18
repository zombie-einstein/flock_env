import chex
import esquilax
import jax.lax
import jax.numpy as jnp

from flock_env import data_types


def exponential_rewards(
    k: chex.PRNGKey, params: data_types.EnvParams, x: chex.Array, y: chex.Array
) -> chex.Array:

    d = esquilax.utils.shortest_distance(x, y, 1.0, norm=False)

    reward = jax.lax.cond(
        d < params.square_min_range,
        lambda _d: -1.0,
        lambda _d: jnp.exp(-40 * jnp.sqrt(_d)),
        d,
    )

    return reward
