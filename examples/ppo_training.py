from typing import Optional, Union

import chex
import jax
import jax.numpy as jnp
import jax_ppo
import optax
import ppo_agent
from esquilax.ml import rl

import flock_env


def training(
    *,
    rng: Union[int, chex.PRNGKey],
    n_agents: int,
    n_train_steps: int,
    test_every: int,
    n_env_steps: int,
    n_train_env: int,
    n_test_env: int,
    n_update_epochs: int,
    mini_batch_size: int,
    max_mini_batches: int,
    training_schedule: Union[float, optax.Schedule],
    network_layer_width: int,
    n_network_layers: int,
    env_params: Optional[flock_env.EnvParams] = None,
    ppo_params: Optional[jax_ppo.PPOParams] = None,
    show_progress: bool = True,
    env_type=flock_env.SimpleFlockEnv,
    **env_kwargs,
):
    """
    Train and test boid PPO policy

    Parameters
    ----------
    rng
        Jax random key or random seed
    n_agents
        Number of boids to simulate
    n_train_steps
        Number of training updates (i.e.
        number of times we sample trajectories)
    test_every
        Number of steps between policy tests
    n_env_steps
        Number of steps to run the environment
        each step
    n_train_env
        Number of environments to simultaneously
        collect samples from each step
    n_test_env
        Number of environments to run simultaneously
        during testing
    n_update_epochs
        Number of policy update within each
        training step
    mini_batch_size
        Mini-batch size to sample each update
    max_mini_batches
        Maximum number of mini-batches to use
    training_schedule
        Either a float learning-rate, or optax
        learning rate schedule
    network_layer_width
        Width of PPO policy-network layers
    n_network_layers
        Number of PPO policy network layers
    env_params
        Optional flock environment parameters. If
        not provided default parameters will be
        used
    ppo_params
        Optional ppo agent parameters. If
        not provided default parameters will be
        used
    show_progress
        If ``True``  tqdm progres bar will be displayed
    env_type
        Environment type to initialise

    Returns
    -------
    tuple
        Tuple containing:

        - Rewards generated over training
        - Training losses
        - Rewards recorded during training
        - Environment states recorded during training
    """
    assert n_train_steps % test_every == 0

    key = rng if isinstance(rng, chex.PRNGKey) else jax.random.PRNGKey(rng)
    k_init, k_train = jax.random.split(key, 2)

    env = env_type(
        reward_func=flock_env.rewards.exponential_rewards,
        n_agents=n_agents,
        **env_kwargs,
    )

    if env_params is None:
        env_params = env.default_params()

    if ppo_params is None:
        ppo_params = jax_ppo.default_params

    obs_shape = env.observation_space(env_params).shape
    n_actions = env.num_actions

    agent = ppo_agent.PPOAgent(
        ppo_params,
        n_update_epochs,
        mini_batch_size,
        max_mini_batches,
    )
    agent_state = agent.init_state(
        k_init,
        obs_shape,
        n_actions,
        training_schedule,
        layer_width=network_layer_width,
        n_layers=n_network_layers,
    )

    n_inner_train = n_train_steps // test_every

    rewards = list()
    losses = list()
    test_rewards = list()
    test_trajectories = list()

    for i in range(n_inner_train):
        k_train, k1, k2 = jax.random.split(k_train, 3)
        agent_state, _train_rewards, _losses = rl.train(
            k1,
            agent,
            agent_state,
            env,
            env_params,
            test_every,
            n_train_env,
            n_env_steps,
            show_progress=show_progress,
        )
        _env_state_records, _test_rewards = rl.test(
            k2,
            agent,
            agent_state,
            env,
            env_params,
            n_test_env,
            n_env_steps,
            show_progress=show_progress,
            return_trajectories=False,
            greedy_actions=True,
        )
        rewards.append(_train_rewards)
        losses.append(_losses)
        test_rewards.append(_test_rewards)
        test_trajectories.append(_env_state_records)

    rewards = jnp.vstack(rewards)
    losses = {
        k: jnp.ravel(jnp.vstack([x[k] for x in losses])) for k in losses[0].keys()
    }
    test_rewards = jnp.stack(test_rewards)

    test_trajectories = flock_env.Boid(
        position=jnp.stack([x.boids.position for x in test_trajectories]),
        speed=jnp.stack([x.boids.speed for x in test_trajectories]),
        heading=jnp.stack([x.boids.heading for x in test_trajectories]),
    )

    return rewards, losses, test_rewards, test_trajectories
