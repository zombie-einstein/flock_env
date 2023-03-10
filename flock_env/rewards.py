import chex
import jax.numpy as jnp

from flock_env import data_types, utils


def exponential_rewards(
    params: data_types.EnvParams, x: chex.Array, x_flock: chex.Array
) -> chex.Array:

    d = utils.shortest_distance(x, x_flock, 1.0, norm=False)
    rewards = jnp.where(d < params.square_range, jnp.exp(-40 * jnp.sqrt(d)), 0.0)
    rewards = jnp.where(d < params.square_min_range, -1.0, rewards)

    return jnp.sum(rewards)
