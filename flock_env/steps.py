import typing
from math import floor
from typing import Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp

from . import data_types


def observe(i_range: float):

    n_bins = floor(1.0 / i_range)

    @esquilax.transforms.spatial(
        n_bins,
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
        i_range=i_range,
    )
    def _observe(
        _k: chex.PRNGKey,
        agent_radius: float,
        a: data_types.Boid,
        b: data_types.Boid,
    ):
        dh = esquilax.utils.shortest_vector(a.heading, b.heading, length=2 * jnp.pi)
        dx = esquilax.utils.shortest_vector(a.position, b.position)
        d = jnp.sqrt(jnp.sum(dx * dx))

        is_close, close_pos = jax.lax.cond(
            d < 2 * agent_radius,
            lambda: (1, dx),
            lambda: (0, jnp.zeros(2)),
        )

        return data_types.Observation(
            n_flock=1,
            pos=dx,
            speed=b.speed,
            heading=dh,
            n_coll=is_close,
            pos_coll=close_pos,
        )

    return _observe


def flatten_observations(i_range: float):
    @esquilax.transforms.amap
    def _flatten_observations(
        _k: chex.PRNGKey,
        params: data_types.EnvParams,
        observations: Tuple[data_types.Boid, data_types.Observation],
    ):
        boid, obs = observations

        def vec_to_polar(dx):
            d = jnp.sqrt(jnp.sum(dx * dx))
            phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
            d_phi = (
                esquilax.utils.shortest_vector(boid.heading, phi, length=2 * jnp.pi)
                / jnp.pi
            )
            return d, d_phi

        def obs_to_nbs():
            _x_nb = obs.pos / obs.n_flock
            _s_nb = obs.speed / obs.n_flock
            _h_nb = obs.heading / obs.n_flock
            d, d_phi = vec_to_polar(_x_nb)
            d = d / i_range
            dh = _h_nb / jnp.pi
            ds = (_s_nb - boid.speed) / (
                params.boids.max_speed - params.boids.min_speed
            )
            return jnp.array([d, d_phi, dh, ds])

        def obs_to_collision():
            _x_close = obs.pos_coll / obs.n_coll
            d, d_phi = vec_to_polar(_x_close)
            d = d / (2 * params.agent_radius)
            return jnp.array([d, d_phi])

        flock_obs = jax.lax.cond(
            obs.n_flock > 0,
            obs_to_nbs,
            lambda: jnp.array([1.0, 0.0, 0.0, 0.0]),
        )
        coll_obs = jax.lax.cond(
            obs.n_coll > 0, obs_to_collision, lambda: jnp.array([1.0, 0.0])
        )

        return jnp.concat([flock_obs, coll_obs])

    return _flatten_observations


@esquilax.transforms.amap
def update_velocity(
    _k: chex.PRNGKey,
    params: data_types.BoidParams,
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
def move(_key: chex.PRNGKey, _params: data_types.BoidParams, x):
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


def rewards(i_range: float):

    n_bins = floor(1.0 / i_range)

    @esquilax.transforms.spatial(
        n_bins,
        (jnp.add, jnp.add),
        (0, 0.0),
        include_self=False,
        topology="moore",
        i_range=i_range,
    )
    def _rewards(
        _k,
        params: data_types.EnvParams,
        boid_a: data_types.Boid,
        boid_b: data_types.Boid,
        *,
        f: typing.Callable
    ):
        d = esquilax.utils.shortest_distance(
            boid_a.position, boid_b.position, norm=True
        )
        reward = f(d / i_range)
        return jax.lax.cond(
            d < 2 * params.agent_radius, lambda: (1, reward), lambda: (0, reward)
        )

    return _rewards


def vision_model(i_range: float, n: int):

    n_bins = floor(1.0 / i_range)

    @esquilax.transforms.spatial(
        n_bins,
        jnp.minimum,
        jnp.ones((n,)),
        include_self=False,
        i_range=i_range,
    )
    def view(
        _k: chex.PRNGKey,
        params: Tuple[float, float],
        a: data_types.Boid,
        b: data_types.Boid,
    ):
        view_angle, radius = params
        rays = jnp.linspace(
            -view_angle * jnp.pi,
            view_angle * jnp.pi,
            n,
            endpoint=True,
        )
        dx = esquilax.utils.shortest_vector(a.position, b.position)
        d = jnp.sqrt(jnp.sum(dx * dx)) / i_range
        phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
        dh = esquilax.utils.shortest_vector(phi, a.heading, 2 * jnp.pi)

        angular_width = jnp.arctan2(radius, d)
        left = dh - angular_width
        right = dh + angular_width

        obs = jnp.where(jnp.logical_and(left < rays, rays < right), d, 1.0)
        return obs

    return view
