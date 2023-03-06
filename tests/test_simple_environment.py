import jax.numpy as jnp
import pytest

import flock_env

N_OBS = 5


@pytest.fixture
def dummy_env():
    def dummy_rewards(*_):
        return jnp.zeros((1,))

    class DummyEnv(flock_env.SimpleFlockEnv):
        def __init__(self, n_agents: int):
            super().__init__(dummy_rewards, n_agents)

    return DummyEnv


@pytest.mark.parametrize(
    "pos, head, expected",
    [
        ([[0.5, 0.5], [0.6, 0.6], [0.4, 0.6]], [0.0, -jnp.pi, jnp.pi], [0.0, 0.1, 0.0]),
        (
            [[0.5, 0.5], [0.6, 0.6], [0.4, 0.6]],
            [0.0, 0.25 * jnp.pi, 0.25 * jnp.pi],
            [0.0, 0.1, 0.25 * jnp.pi],
        ),
        (
            [[0.5, 0.5], [0.6, 0.6], [0.9, 0.9]],
            [0.0, jnp.pi / 2, jnp.pi],
            [0.1, 0.1, jnp.pi / 2],
        ),
        (
            [[0.95, 0.5], [0.05, 0.5], [0.16, 0.5]],
            [0.0, jnp.pi / 2, jnp.pi],
            [0.1, 0.0, jnp.pi / 2],
        ),
        (
            [[0.5, 0.05], [0.5, 0.95], [0.5, 0.5]],
            [0.0, jnp.pi / 2, jnp.pi],
            [0.0, -0.1, jnp.pi / 2],
        ),
        (
            [[0.5, 0.05], [0.1, 0.1], [0.9, 0.9]],
            [0.0, -jnp.pi, jnp.pi],
            [0.0, 0.0, 0.0],
        ),
    ],
)
def test_observation(dummy_env, pos, head, expected):
    env = dummy_env(3)
    r = 0.2
    s = flock_env.EnvState(
        agent_positions=jnp.array(pos),
        agent_speeds=jnp.array([0.0, 0.0, 0.0]),
        agent_headings=jnp.array(head),
        time=jnp.array(0),
    )
    p = flock_env.EnvParams(
        min_speed=0.0,
        max_speed=0.5,
        max_rotate=0.5,
        max_accelerate=0.1,
        square_range=r**2,
        square_min_range=0.0001,
    )

    obs = env.get_obs(s, p)

    assert obs.shape == (3, 3)
    expected = jnp.array([1 / r, 1 / r, 1 / jnp.pi]) * jnp.array(expected)
    assert jnp.all(jnp.isclose(obs[0], expected, atol=1e-6))
