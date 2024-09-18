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
            super().__init__(dummy_rewards, n_agents)

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

    assert env_state.agent_positions.shape == (N_AGENTS, 2)
    assert env_state.agent_headings.shape == (N_AGENTS,)
    assert env_state.agent_speeds.shape == (N_AGENTS,)
    assert env_state.time.shape == (N_AGENTS,)
    assert jnp.array_equal(env_state.time, jnp.zeros((N_AGENTS,), dtype=jnp.int32))
    assert obs.shape == (N_AGENTS, N_OBS)

    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.agent_positions, 0.0),
            jnp.less_equal(env_state.agent_positions, 1.0),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.agent_headings, 0.0),
            jnp.less_equal(env_state.agent_headings, 2 * jnp.pi),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(env_state.agent_speeds, params.min_speed),
            jnp.less_equal(env_state.agent_speeds, params.max_speed),
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

    assert state_seq.agent_positions.shape == (n_steps, N_AGENTS, 2)
    assert state_seq.agent_headings.shape == (n_steps, N_AGENTS)
    assert state_seq.agent_speeds.shape == (n_steps, N_AGENTS)
    assert obs_seq.shape == (n_steps, N_AGENTS, N_OBS)

    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.agent_positions, 0.0),
            jnp.less_equal(state_seq.agent_positions, 1.0),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.agent_headings, 0.0),
            jnp.less_equal(state_seq.agent_headings, 2 * jnp.pi),
        )
    )
    assert jnp.all(
        jnp.logical_and(
            jnp.greater_equal(state_seq.agent_speeds, params.min_speed),
            jnp.less_equal(state_seq.agent_speeds, params.max_speed),
        )
    )
    assert jnp.array_equal(state_seq.time[:, 0], jnp.arange(n_steps) + 1)


def test_movement(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        agent_positions=jnp.array([[0.5, 0.5]]),
        agent_speeds=jnp.array([0.25]),
        agent_headings=jnp.array([jnp.pi]),
        time=jnp.array([0]),
    )
    p = flock_env.EnvParams(
        min_speed=0.1,
        max_speed=0.25,
        max_rotate=0.5,
        max_accelerate=0.1,
        square_min_range=0.0001,
        collision_penalty=0.1,
    )

    actions = jnp.zeros((1, 2))

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.agent_positions

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
        s0.replace(
            agent_headings=jnp.array([2 * jnp.pi]),
        ),
        jnp.array(
            [[[0.5, 0.5]], [[0.75, 0.5]], [[0.0, 0.5]], [[0.25, 0.5]], [[0.5, 0.5]]]
        ),
    )

    sub_test(
        s0.replace(
            agent_headings=jnp.array([0.5 * jnp.pi]),
        ),
        jnp.array(
            [[[0.5, 0.5]], [[0.5, 0.75]], [[0.5, 0.0]], [[0.5, 0.25]], [[0.5, 0.5]]]
        ),
    )


def test_rotation(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        agent_positions=jnp.array([[0.5, 0.5]]),
        agent_speeds=jnp.array([0.25]),
        agent_headings=jnp.array([jnp.pi]),
        time=jnp.array([0]),
    )
    p = flock_env.EnvParams(
        min_speed=0.1,
        max_speed=0.25,
        max_rotate=0.5,
        max_accelerate=0.1,
        square_min_range=0.0001,
        collision_penalty=0.1,
    )

    actions = jnp.array([[1.0, 0.0]])

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.agent_positions

    _, pos_seq = jax.lax.scan(step, s0, None, length=5)
    expected = jnp.array(
        [[[0.5, 0.5]], [[0.5, 0.25]], [[0.75, 0.25]], [[0.75, 0.5]], [[0.5, 0.5]]]
    )

    assert jnp.all(jnp.isclose(pos_seq, expected))


def test_acceleration(dummy_env):
    test_env = dummy_env(1)

    k = jax.random.PRNGKey(101)
    s0 = flock_env.EnvState(
        agent_positions=jnp.array([[0.0, 0.5]]),
        agent_speeds=jnp.array([0.0]),
        agent_headings=jnp.array([0.0]),
        time=jnp.array([0]),
    )
    p = flock_env.EnvParams(
        min_speed=0.0,
        max_speed=0.5,
        max_rotate=0.5,
        max_accelerate=0.1,
        square_min_range=0.0001,
        collision_penalty=0.1,
    )

    actions = jnp.array([[0.0, 1.0]])

    def step(s, _):
        _, new_state, _, _ = test_env.step(k, p, s, actions)
        return new_state, s.agent_positions

    _, pos_seq = jax.lax.scan(step, s0, None, length=5)
    expected = jnp.array(
        [[[0.0, 0.5]], [[0.1, 0.5]], [[0.3, 0.5]], [[0.6, 0.5]], [[0.0, 0.5]]]
    )

    assert jnp.all(jnp.isclose(pos_seq, expected))
