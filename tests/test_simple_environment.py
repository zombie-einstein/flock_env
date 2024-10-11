import jax.numpy as jnp
import jax.random
import pytest

import flock_env

N_OBS = 5


@pytest.fixture
def dummy_env():
    def dummy_rewards(*_):
        return 1.0

    class DummyEnv(flock_env.SimpleFlockEnv):
        def __init__(self, n_agents: int):
            super().__init__(dummy_rewards, n_agents, 0.1)

    return DummyEnv


@pytest.mark.parametrize(
    "pos, head, expected",
    [
        (
            [[0.5, 0.5], [0.5, 0.55], [0.4, 0.6]],
            [0.0, -jnp.pi, jnp.pi],
            [0.5, 0.5 * jnp.pi, jnp.pi, 0.0],
        ),
        (
            [[0.5, 0.5], [0.5, 0.55], [0.4, 0.6]],
            [0.0, 0.25 * jnp.pi, 0.25 * jnp.pi],
            [0.5, 0.5 * jnp.pi, 0.25 * jnp.pi, 0.0],
        ),
        (
            [[0.5, 0.05], [0.1, 0.1], [0.9, 0.9]],
            [0.0, -jnp.pi, jnp.pi],
            [1.0, 0.0, 0.0, 0.0],
        ),
    ],
)
def test_observation(dummy_env, pos, head, expected):
    k = jax.random.PRNGKey(101)
    n_agents = 3
    env = dummy_env(n_agents)
    n_obs = env.observation_space(env.default_params()).shape[0]
    s = flock_env.Boid(
        position=jnp.array(pos),
        speed=jnp.array([0.0, 0.0, 0.0]),
        heading=jnp.array(head),
    )
    p = flock_env.EnvParams(
        boids=flock_env.BoidParams(
            min_speed=0.0,
            max_speed=0.5,
            max_rotate=0.5,
            max_accelerate=0.1,
        ),
        agent_radius=0.01,
        collision_penalty=0.1,
    )

    obs = env.get_obs(k, p, s)

    assert obs.shape == (n_agents, n_obs)
    expected = jnp.array([1, 1 / jnp.pi, 1 / jnp.pi, 1.0]) * jnp.array(expected)
    assert jnp.all(jnp.isclose(obs[0, :4], expected, atol=1e-6))
