from typing import Dict

import numpy as np
import pytest

from flock_env import DiscreteActionFlock

test_environments = [
    {
        "n_agents": 3,
        "speed": 0.1,
        "n_steps": 200,
        "rotation_size": 0.2,
        "n_actions": 3,
        "n_nearest": 2,
    },
    {
        "n_agents": 3,
        "speed": 0.2,
        "n_steps": 100,
        "rotation_size": 0.1,
        "n_actions": 5,
        "n_nearest": 2,
    },
    {
        "n_agents": 10,
        "speed": 0.2,
        "n_steps": 1000,
        "rotation_size": 0.01,
        "n_actions": 7,
        "n_nearest": 3,
    },
    {
        "n_agents": 10,
        "speed": 0.2,
        "n_steps": 1000,
        "rotation_size": 0.01,
        "n_actions": 7,
        "n_nearest": 3,
        "obstacles": ((0.0, 0.0, 0.5),),
    },
]


@pytest.mark.parametrize("params", test_environments)
def test_environment_initialization(params: Dict):
    env = DiscreteActionFlock(**params)
    n_obstacles = len(params.get("obstacles", ()))
    assert env.x.shape == (2, params["n_agents"] + n_obstacles)
    assert env.rotations.shape[0] == params["n_actions"]
    for i in range((params["n_actions"] - 1) // 2):
        assert -env.rotations[i] == env.rotations[-(i + 1)]


@pytest.mark.parametrize("params", test_environments)
def test_environment_reset(params: Dict):
    env = DiscreteActionFlock(**params)
    obstacle_x = env.x[:, params["n_agents"] :].copy()
    n_obstacles = len(params.get("obstacles", ()))
    for _ in range(100):
        obs = env.reset()
        assert obs.shape == (
            params["n_agents"],
            3 * params["n_nearest"] + (2 * n_obstacles),
        )
        assert np.array_equal(obstacle_x, env.x[:, params["n_agents"] :])
        assert ((-1.0 <= obs) & (obs <= 1.0)).all()
        assert env.i == 0
        assert ((env.x >= 0) & (env.x <= 1.0)).all()
        assert ((env.speed >= 0) & (env.speed <= 1.0)).all()
        assert ((env.theta >= 0) & (env.theta <= 2 * np.pi)).all()


@pytest.mark.parametrize("params", test_environments)
def test_agent_update(params: Dict):
    n_agents = params["n_agents"]

    env = DiscreteActionFlock(**params)

    # Manually set phase space values
    env.theta = np.array([0 for _ in range(n_agents)]).astype(np.float32)
    env.speed = np.array([0.1 for _ in range(n_agents)]).astype(np.float32)
    env.x[:, :n_agents] = np.array(
        [[0.5 for _ in range(n_agents)], [0.5 for _ in range(n_agents)]]
    ).astype(np.float32)

    for _ in range(10):
        old_x = env.x.copy()
        env._update_agents()
        assert np.allclose(
            env.x[0, :n_agents], old_x[0, :n_agents] + env.speed * params["speed"],
        )
        assert np.allclose(env.x[1, :n_agents], old_x[1, :n_agents])
        assert np.allclose(env.x[:, n_agents:], old_x[:, n_agents:])

    env.reset()

    for _ in range(1000):
        env._update_agents()
        assert ((env.x >= 0) & (env.x <= 1.0)).all()


@pytest.mark.parametrize("params", test_environments)
def test_step(params: Dict):
    env = DiscreteActionFlock(**params)
    env.reset()
    done = False

    n_nearest = params["n_nearest"]
    n_agents = params["n_agents"]
    n_obstacles = len(params.get("obstacles", ()))

    while not done:
        actions = np.random.randint(0, params["n_actions"], params["n_agents"])
        obs, rewards, done, debug = env.step(actions)
        assert obs.shape == (n_agents, n_nearest * 3 + 2 * n_obstacles)
        assert ((obs >= -1.0) & (obs <= 1.0)).all()
        assert ((env.x >= 0) & (env.x <= 1.0)).all()
        assert ((env.speed >= 0) & (env.speed <= 1.0)).all()
        assert ((env.theta >= 0) & (env.theta <= 2 * np.pi)).all()
