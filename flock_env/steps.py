from typing import Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp

from . import data_types


@esquilax.transforms.spatial(
    10,
    data_types.Observation(
        n_flock=jnp.add,
        pos=jnp.add,
        speed=jnp.add,
        heading=jnp.add,
        n_coll=jnp.add,
        pos_coll=jnp.add,
    ),
    data_types.Observation(),
    include_self=False,
)
def observe(
    _k: chex.PRNGKey,
    params: data_types.EnvParams,
    a: data_types.Boid,
    b: data_types.Boid,
):
    dh = esquilax.utils.shortest_vector(a.heading, b.heading, length=2 * jnp.pi)
    d = esquilax.utils.shortest_distance(a.position, b.position, norm=True)
    is_close, close_pos = jax.lax.cond(
        d < 2 * params.agent_radius,
        lambda: (1, b.position),
        lambda: (0, jnp.zeros(2)),
    )

    return data_types.Observation(
        n_flock=1,
        pos=b.position,
        speed=b.speed,
        heading=dh,
        n_coll=is_close,
        pos_coll=close_pos,
    )


@esquilax.transforms.amap
def flatten_observations(
    _k: chex.PRNGKey,
    params: data_types.EnvParams,
    observations: Tuple[data_types.Boid, data_types.Observation],
):
    boid, obs = observations

    def vec_to_polar(dx):
        d = jnp.sqrt(jnp.sum(dx * dx)) / 0.1
        phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
        d_phi = (
            esquilax.utils.shortest_vector(boid.heading, phi, length=2 * jnp.pi)
            / jnp.pi
        )
        return d, d_phi

    def obs_to_nbs():
        _x_nb = obs.pos / obs.n_flock
        _s_nb = obs.speed / obs.n_flock
        _h_nb = obs.heading / obs.n_flock
        dx = esquilax.utils.shortest_vector(boid.position, _x_nb)
        d, d_phi = vec_to_polar(dx)
        dh = _h_nb / jnp.pi
        ds = (_s_nb - boid.speed) / (params.max_speed - params.min_speed)

        return jnp.array([d, d_phi, dh, ds])

    def obs_to_collision():
        _x_close = obs.pos_coll / obs.n_coll
        dx = esquilax.utils.shortest_vector(boid.position, _x_close)
        d, d_phi = vec_to_polar(dx)
        return jnp.array([d, d_phi])

    flock_obs = jax.lax.cond(
        obs.n_flock > 0,
        obs_to_nbs,
        lambda: jnp.array([-1.0, 0.0, 0.0, -1.0]),
    )
    coll_obs = jax.lax.cond(
        obs.n_coll > 0, obs_to_collision, lambda: jnp.array([-1.0, 0.0])
    )

    return jnp.concat([flock_obs, coll_obs])


@esquilax.transforms.amap
def update_velocity(
    _k: chex.PRNGKey,
    params: data_types.EnvParams,
    x: Tuple[chex.Array, data_types.Boid],
):
    actions, boid = x
    rotation = actions[0] * params.max_rotate * jnp.pi
    acceleration = actions[1] * params.max_accelerate

    new_heading = (boid.heading + rotation) % (2 * jnp.pi)
    new_speeds = jnp.clip(
        boid.speed + acceleration,
        min=params.min_speed,
        max=params.max_speed,
    )

    return new_heading, new_speeds


@esquilax.transforms.amap
def move(_key: chex.PRNGKey, _params: data_types.EnvParams, x):
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


def vision_model(n: int, n_bins=10):
    @esquilax.transforms.spatial(
        n_bins,
        jnp.minimum,
        jnp.ones((n,)),
        include_self=False,
    )
    def view(
        _k: chex.PRNGKey,
        params: data_types.EnvParams,
        a: data_types.Boid,
        b: data_types.Boid,
    ):
        rays = jnp.linspace(
            -params.view_angle * jnp.pi,
            params.view_angle * jnp.pi,
            n,
            endpoint=True,
        )
        dx = esquilax.utils.shortest_vector(a.position, b.position)
        d = jnp.sqrt(jnp.sum(dx * dx))
        phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
        dh = esquilax.utils.shortest_vector(phi, a.heading, 2 * jnp.pi)

        angular_width = jnp.arctan2(params.agent_radius, d)
        left = dh - angular_width
        right = dh + angular_width

        obs = jnp.where(jnp.logical_and(left < rays, rays < right), n_bins * d, 1.0)
        return obs

    return view
