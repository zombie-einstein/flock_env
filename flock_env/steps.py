from typing import Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp

from .data_types import EnvParams, EnvState


@esquilax.transforms.spatial(
    10,
    (jnp.add, jnp.add, jnp.add, jnp.add),
    (0, jnp.zeros(2), 0.0, 0.0),
    include_self=False,
)
def observe(_k: chex.PRNGKey, _params: EnvParams, a: EnvState, b: EnvState):
    dh = esquilax.utils.shortest_vector(
        a.agent_headings, b.agent_headings, length=2 * jnp.pi
    )
    return 1, b.agent_positions, b.agent_speeds, dh


@esquilax.transforms.amap
def flatten_observations(_k: chex.PRNGKey, params: EnvParams, observations):
    boid, n_nb, x_nb, s_nb, h_nb = observations

    def obs_to_nbs():
        _x_nb = x_nb / n_nb
        _s_nb = s_nb / n_nb
        _h_nb = h_nb / n_nb

        dx = esquilax.utils.shortest_vector(boid.agent_positions, _x_nb)

        d = jnp.sqrt(jnp.sum(dx * dx)) / 0.1

        phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
        d_phi = (
            esquilax.utils.shortest_vector(boid.agent_headings, phi, 2 * jnp.pi)
            / jnp.pi
        )

        dh = _h_nb / jnp.pi
        ds = (_s_nb - boid.agent_speeds) / (params.max_speed - params.min_speed)

        return jnp.array([d, d_phi, dh, ds])

    return jax.lax.cond(
        n_nb > 0,
        obs_to_nbs,
        lambda: jnp.array([-1.0, 0.0, 0.0, -1.0]),
    )


@esquilax.transforms.amap
def update_velocity(
    _k: chex.PRNGKey, params: EnvParams, x: Tuple[chex.Array, EnvState]
):
    actions, boid = x
    rotation = actions[0] * params.max_rotate * jnp.pi
    acceleration = actions[1] * params.max_accelerate

    new_heading = (boid.agent_headings + rotation) % (2 * jnp.pi)
    new_speeds = jnp.clip(
        boid.agent_speeds + acceleration,
        min=params.min_speed,
        max=params.max_speed,
    )

    return new_heading, new_speeds


@esquilax.transforms.amap
def move(_key: chex.PRNGKey, _params: EnvParams, x):
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0
