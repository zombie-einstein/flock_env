import chex
import jax

from . import steps
from .data_types import Boid, BoidParams


def init_boids(n: int, params: BoidParams, key: chex.PRNGKey) -> Boid:
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.uniform(k1, (n, 2))
    speeds = jax.random.uniform(
        k2, (n,), minval=params.min_speed, maxval=params.max_speed
    )
    headings = jax.random.uniform(k3, (n,), minval=0.0, maxval=2.0 * jax.numpy.pi)

    return Boid(
        position=positions,
        speed=speeds,
        heading=headings,
    )


def update_state(
    key: chex.PRNGKey, params: BoidParams, boids: Boid, actions: chex.Array
) -> Boid:
    actions = jax.numpy.clip(actions, min=-1.0, max=1.0)
    headings, speeds = steps.update_velocity(key, params, (actions, boids))
    positions = steps.move(key, params, (boids.position, headings, speeds))

    return Boid(
        position=positions,
        speed=speeds,
        heading=headings,
    )
