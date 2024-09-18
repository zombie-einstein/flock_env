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
            super().__init__(dummy_rewards, n_agents)

    return DummyEnv


@pytest.mark.parametrize(
    "pos, head, expected",
    [
        (
            [[0.5, 0.5], [0.5, 0.55], [0.4, 0.6]],
            [0.0, -jnp.pi, jnp.pi],
            [0.5, -0.5 * jnp.pi, jnp.pi, 0.0],
        ),
        (
            [[0.5, 0.5], [0.5, 0.55], [0.4, 0.6]],
            [0.0, 0.25 * jnp.pi, 0.25 * jnp.pi],
            [0.5, -0.5 * jnp.pi, 0.25 * jnp.pi, 0.0],
        ),
        (
            [[0.5, 0.05], [0.1, 0.1], [0.9, 0.9]],
            [0.0, -jnp.pi, jnp.pi],
            [-1.0, 0.0, 0.0, -1.0],
        ),
    ],
)
def test_observation(dummy_env, pos, head, expected):
    k = jax.random.PRNGKey(101)
    env = dummy_env(3)
    s = flock_env.EnvState(
        agent_positions=jnp.array(pos),
        agent_speeds=jnp.array([0.0, 0.0, 0.0]),
        agent_headings=jnp.array(head),
        time=jnp.array([0, 0, 0]),
    )
    p = flock_env.EnvParams(
        min_speed=0.0,
        max_speed=0.5,
        max_rotate=0.5,
        max_accelerate=0.1,
        square_min_range=0.0001,
        collision_penalty=0.1,
    )

    obs = env.get_obs(k, p, s)

    assert obs.shape == (3, 4)
    expected = jnp.array([1, 1 / jnp.pi, 1 / jnp.pi, 1.0]) * jnp.array(expected)
    assert jnp.all(jnp.isclose(obs[0], expected, atol=1e-6))
