import jax
import pytest

import flock_env


@pytest.mark.parametrize("sparse_rewards", [True, False])
def test_pred_prey_env(sparse_rewards):

    k = jax.random.PRNGKey(101)
    n_vision = 10
    n_predators = 2
    n_prey = 12

    env = flock_env.BasePredatorPreyEnv(
        n_predators=n_predators,
        n_prey=n_prey,
        prey_vision_range=0.2,
        predator_vision_range=0.4,
        n_vision=n_vision,
        agent_radius=0.01,
        sparse_rewards=sparse_rewards,
    )

    params = env.default_params()
    obs, state = env.reset(k, params)

    assert isinstance(obs, flock_env.PredatorPrey)
    assert obs.predator.shape == (
        n_predators,
        2 * n_vision,
    )
    assert obs.prey.shape == (
        n_prey,
        2 * n_vision,
    )

    k1, k2 = jax.random.split(k, 2)
    actions = flock_env.PredatorPrey(
        predator=jax.random.uniform(k1, (n_predators, 2)),
        prey=jax.random.uniform(k1, (n_prey, 2)),
    )

    obs, state, rewards, done = env.step(k, params, state, actions)

    assert isinstance(obs, flock_env.PredatorPrey)
    assert obs.predator.shape == (
        n_predators,
        2 * n_vision,
    )
    assert obs.prey.shape == (
        n_prey,
        2 * n_vision,
    )

    assert isinstance(state, flock_env.PredatorPreyState)
    assert isinstance(state.predators, flock_env.Boid)
    assert state.predators.position.shape == (n_predators, 2)
    assert state.predators.heading.shape == (n_predators,)
    assert state.predators.speed.shape == (n_predators,)

    assert isinstance(state.prey, flock_env.Boid)
    assert state.prey.position.shape == (n_prey, 2)
    assert state.prey.heading.shape == (n_prey,)
    assert state.prey.speed.shape == (n_prey,)

    assert isinstance(rewards, flock_env.PredatorPrey)
    assert rewards.predator.shape == (n_predators,)
    assert rewards.prey.shape == (n_prey,)

    assert isinstance(done, flock_env.PredatorPrey)
    assert done.predator.shape == (n_predators,)
    assert done.prey.shape == (n_prey,)
