import chex
import jax
import jax.numpy as jnp
import pytest
from gymnax.environments import spaces

import flock_env
from flock_env.base_env import BaseFlockEnv

N_AGENTS = 101
N_OBS = 5


@pytest.fixture
def dummy_env():
    def dummy_rewards(*_):
        return 0

    class DummyEnv(BaseFlockEnv):
        def __init__(self, n_agents: int):
            super().__init__(dummy_rewards, n_agents, 0.1)

        def get_obs(
            self,
            key: chex.PRNGKey,
            params: flock_env.EnvParams,
            state: flock_env.EnvState,
        ) -> chex.Array:
            return jnp.zeros((self.n_agents, N_OBS))

        def observation_space(self, params: flock_env.EnvParams):
            return spaces.Box(-1.0, 1.0, shape=(N_OBS,), dtype=jnp.float32)

    return DummyEnv


@pytest.fixture
def env(dummy_env):
    return dummy_env(N_AGENTS)


@pytest.fixture
def params(env: BaseFlockEnv):
    return env.default_params()


def test_reset(params: flock_env.EnvParams, env: BaseFlockEnv):
    k = jax.random.PRNGKey(101)
    obs, env_state = env.reset(k, params)

    assert isinstance(env_state, flock_env.EnvState)

    assert env_state.boids.position.shape == (N_AGENTS, 2)
    assert env_state.boids.heading.shape == (N_AGENTS,)
    assert env_state.boids.speed.shape == (N_AGENTS,)
    assert env_state.step == 0
    assert obs.shape == (N_AGENTS, N_OBS)

    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.boids.position, 0.0),
            jnp.less_equal(env_state.boids.position, 1.0),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.boids.heading, 0.0),
            jnp.less_equal(env_state.boids.heading, 2 * jnp.pi),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.boids.speed, params.boids.min_speed),
            jnp.less_equal(env_state.boids.speed, params.boids.max_speed),
        )
    )


def test_update_sequence(params: flock_env.EnvParams, env: BaseFlockEnv):
    def step(carry, _):
        k, state = carry
        k1, k2, k3 = jax.random.split(k, num=3)
        actions = jax.random.uniform(k1, (N_AGENTS, 2))
        new_obs, new_state, rewards, dones = env.step(
            k2,
            params,
            state,
            actions,
        )
        return (k3, new_state), (new_obs, new_state, rewards, dones)

    n_steps = 200

    key = jax.random.PRNGKey(101)
    _, state_0 = env.reset(key, params)

    _, (obs_seq, state_seq, reward_seq, dones_seq) = jax.lax.scan(
        step, (key, state_0), None, length=n_steps
    )

    assert isinstance(state_seq, flock_env.EnvState)

    assert state_seq.boids.position.shape == (n_steps, N_AGENTS, 2)
    assert state_seq.boids.heading.shape == (n_steps, N_AGENTS)
    assert state_seq.boids.speed.shape == (n_steps, N_AGENTS)
    assert obs_seq.shape == (n_steps, N_AGENTS, N_OBS)

    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.boids.position, 0.0),
            jnp.less_equal(state_seq.boids.position, 1.0),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.boids.heading, 0.0),
            jnp.less_equal(state_seq.boids.heading, 2 * jnp.pi),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.boids.speed, params.boids.min_speed),
            jnp.less_equal(state_seq.boids.speed, params.boids.max_speed),
        )
    )
    assert jnp.array_equal(state_seq.step, jnp.arange(n_steps) + 1)


def test_movement(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        boids=flock_env.Boid(
            position=jnp.array([[0.5, 0.5]]),
            speed=jnp.array([0.25]),
            heading=jnp.array([jnp.pi]),
        ),
        step=0,
    )
    p = flock_env.EnvParams(
        boids=flock_env.BoidParams(
            min_speed=0.1,
            max_speed=0.25,
            max_rotate=0.5,
            max_accelerate=0.1,
        ),
        agent_radius=0.01,
        collision_penalty=0.1,
    )

    actions = jnp.zeros((1, 2))

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.boids.position

    def sub_test(s, expected):
        _, pos_seq = jax.lax.scan(step, s, None, length=5)
        assert jnp.all(jnp.isclose(pos_seq, expected))

    sub_test(
        s0,
        jnp.array(
            [[[0.5, 0.5]], [[0.25, 0.5]], [[0.0, 0.5]], [[0.75, 0.5]], [[0.5, 0.5]]]
        ),
    )

    sub_test(
        s0.replace(boids=s0.boids.replace(heading=jnp.array([2 * jnp.pi]))),
        jnp.array(
            [[[0.5, 0.5]], [[0.75, 0.5]], [[0.0, 0.5]], [[0.25, 0.5]], [[0.5, 0.5]]]
        ),
    )

    sub_test(
        s0.replace(
            boids=s0.boids.replace(
                heading=jnp.array([0.5 * jnp.pi]),
            )
        ),
        jnp.array(
            [[[0.5, 0.5]], [[0.5, 0.75]], [[0.5, 0.0]], [[0.5, 0.25]], [[0.5, 0.5]]]
        ),
    )


def test_rotation(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        boids=flock_env.Boid(
            position=jnp.array([[0.5, 0.5]]),
            speed=jnp.array([0.25]),
            heading=jnp.array([jnp.pi]),
        ),
        step=0,
    )
    p = flock_env.EnvParams(
        boids=flock_env.BoidParams(
            min_speed=0.1,
            max_speed=0.25,
            max_rotate=0.5,
            max_accelerate=0.1,
        ),
        agent_radius=0.01,
        collision_penalty=0.1,
    )

    actions = jnp.array([[1.0, 0.0]])

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.boids.position

    _, pos_seq = jax.lax.scan(step, s0, None, length=5)
    expected = jnp.array(
        [[[0.5, 0.5]], [[0.5, 0.25]], [[0.75, 0.25]], [[0.75, 0.5]], [[0.5, 0.5]]]
    )

    assert jnp.all(jnp.isclose(pos_seq, expected))


def test_acceleration(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        boids=flock_env.Boid(
            position=jnp.array([[0.0, 0.5]]),
            speed=jnp.array([0.0]),
            heading=jnp.array([0.0]),
        ),
        step=0,
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

    actions = jnp.array([[0.0, 1.0]])

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.boids.position

    _, pos_seq = jax.lax.scan(step, s0, None, length=5)
    expected = jnp.array(
        [[[0.0, 0.5]], [[0.1, 0.5]], [[0.3, 0.5]], [[0.6, 0.5]], [[0.0, 0.5]]]
    )

    assert jnp.all(jnp.isclose(pos_seq, expected))
