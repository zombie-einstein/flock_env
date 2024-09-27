import jax.numpy as jnp


def exponential_rewards(d: float) -> float:
    return jnp.exp(-5 * d)
