import jax.numpy as jnp


def shortest_vector(a, b, length):
    x = b - a
    x_ = jnp.sign(x) * (jnp.abs(x) - length)
    return jnp.where(jnp.abs(x) < jnp.abs(x_), x, x_)


def shortest_distance(a, b, length, norm=True):
    x = jnp.abs(a - b)
    d = jnp.where(x > 0.5 * length, length - x, x)
    d = jnp.sum(jnp.square(d), axis=-1)
    if norm:
        d = jnp.sqrt(d)
    return d
