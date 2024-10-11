from typing import Optional, Union

import chex
import jax
import jax_ppo
import optax
import ppo_agent
from esquilax.ml import rl

import flock_env


def training(
    *,
    rng: Union[int, chex.PRNGKey],
    n_predators: int,
    n_prey: int,
    predator_vision_range: float,
    prey_vision_range: float,
    n_vision: int,
    agent_radius: float,
    sparse_rewards: bool,
    n_train_steps: int,
    test_every: int,
    n_env_steps: int,
    n_train_env: int,
    n_test_env: int,
    n_update_epochs: int,
    mini_batch_size: int,
    max_mini_batches: int,
    predator_training_schedule: Union[float, optax.Schedule],
    prey_training_schedule: Union[float, optax.Schedule],
    network_layer_width: int,
    n_network_layers: int,
    env_params: Optional[flock_env.PredatorPreyParams] = None,
    ppo_params: Optional[jax_ppo.PPOParams] = None,
    show_progress: bool = True,
    return_trajectories: bool = False,
):
    assert n_train_steps % test_every == 0

    key = rng if isinstance(rng, chex.PRNGKey) else jax.random.PRNGKey(rng)
    k_init_predator, k_init_prey, k_train = jax.random.split(key, 3)

    env = flock_env.BasePredatorPreyEnv(
        n_predators=n_predators,
        n_prey=n_prey,
        prey_vision_range=prey_vision_range,
        predator_vision_range=predator_vision_range,
        n_vision=n_vision,
        agent_radius=agent_radius,
        sparse_rewards=sparse_rewards,
    )

    if env_params is None:
        env_params = env.default_params()

    if ppo_params is None:
        ppo_params = jax_ppo.default_params

    obs_shape = env.observation_space(env_params).shape
    n_actions = env.num_actions

    predator_agent = ppo_agent.PPOAgent(
        ppo_params,
        n_update_epochs,
        mini_batch_size,
        max_mini_batches,
    )
    predator_state = predator_agent.init_state(
        k_init_predator,
        obs_shape,
        n_actions,
        predator_training_schedule,
        layer_width=network_layer_width,
        n_layers=n_network_layers,
    )

    prey_agent = ppo_agent.PPOAgent(
        ppo_params,
        n_update_epochs,
        mini_batch_size,
        max_mini_batches,
    )
    prey_state = prey_agent.init_state(
        k_init_prey,
        obs_shape,
        n_actions,
        prey_training_schedule,
        layer_width=network_layer_width,
        n_layers=n_network_layers,
    )

    agents = flock_env.PredatorPrey(
        predator=predator_agent,
        prey=prey_agent,
    )
    agent_states = flock_env.PredatorPrey(
        predator=predator_state,
        prey=prey_state,
    )

    _, rewards, losses, test_trajectories, test_rewards = rl.train_and_test(
        k_train,
        agents,
        agent_states,
        env,
        env_params,
        n_train_steps,
        test_every,
        n_train_env,
        n_test_env,
        n_env_steps,
        show_progress=show_progress,
        return_trajectories=return_trajectories,
        greedy_test_actions=True,
    )

    return rewards, losses, test_trajectories, test_rewards
