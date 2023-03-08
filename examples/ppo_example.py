import jax
import jax_ppo
import optax

import flock_env


def flock_training_runner(
    n_agents=20,
    n_train=2_000,
    n_train_env=32,
    n_test_env=5,
    n_env_steps=200,
    n_train_epochs=2,
    mini_batch_size=256,
):
    k = jax.random.PRNGKey(101)
    n_steps = n_train * n_train_env * n_env_steps * n_train_epochs // mini_batch_size

    params = jax_ppo.default_params._replace(
        gamma=0.95, gae_lambda=0.9, entropy_coeff=0.0001, adam_eps=1e-8, clip_coeff=0.2
    )

    train_schedule = optax.linear_schedule(2e-3, 2e-5, n_steps)

    env = flock_env.SimpleFlockEnv(
        reward_func=flock_env.rewards.exponential_rewards, n_agents=n_agents
    )
    env_params = env.default_params

    k, agent = jax_ppo.init_agent(
        k,
        params,
        env.action_space(env_params).shape,
        env.observation_space(env_params).shape,
        train_schedule,
        layer_width=16,
        n_agents=n_agents,
    )

    k, trained_agent, losses, ts, rewards = jax_ppo.train(
        k,
        env,
        env_params,
        agent,
        n_train,
        n_train_env,
        n_train_epochs,
        mini_batch_size,
        n_test_env,
        params,
        greedy_test_policy=True,
        n_env_steps=n_env_steps,
        n_agents=n_agents,
    )


if __name__ == "__main__":
    flock_training_runner()
