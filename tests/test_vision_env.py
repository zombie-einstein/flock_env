import jax
import jax.numpy as jnp
import pytest

import flock_env

N_VIEW = 5
N_AGENTS = 2


@pytest.mark.parametrize(
    "pos, heading, expected",
    [
        (
            [[0.5, 0.5], [0.45, 0.5]],
            0.0,
            jnp.ones(
                N_VIEW,
            ),
        ),
        (
            [[0.5, 0.5], [0.55, 0.5]],
            0.0,
            jnp.array([1.0, 1.0, 0.5, 1.0, 1.0]),
        ),
        (
            [[0.5, 0.5], [0.5, 0.55]],
            0.0,
            jnp.array([0.5, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            [[0.5, 0.5], [0.5, 0.45]],
            0.0,
            jnp.array([1.0, 1.0, 1.0, 1.0, 0.5]),
        ),
        (
            [[0.5, 0.5], [0.55, 0.5]],
            0.5 * jnp.pi,
            jnp.array([1.0, 1.0, 1.0, 1.0, 0.5]),
        ),
        (
            [[0.5, 0.5], [0.45, 0.5]],
            0.5 * jnp.pi,
            jnp.array([0.5, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            [[0.5, 0.5], [0.55, 0.55]],
            0.5 * jnp.pi,
            jnp.array([1.0, 1.0, 1.0, 0.7071069, 1.0]),
        ),
    ],
)
def test_observation(pos, heading, expected):

    k = jax.random.PRNGKey(101)
    env = flock_env.VisionEnv(
        lambda *_: 10.0,
        N_AGENTS,
        n_view=N_VIEW,
    )
    params = env.default_params()

    boids = flock_env.Boid(
        position=jnp.array(pos),
        speed=jnp.zeros((2,)),
        heading=jnp.array([heading, 0]),
    )

    obs = env.get_obs(k, params, boids)

    assert obs.shape == (N_AGENTS, N_VIEW)
    assert jnp.allclose(obs[0], expected)
