def test_example():
    import jax

    import flock_env

    key = jax.random.PRNGKey(101)
    key_reset, key_act, key_step = jax.random.split(key, 3)

    # Initialise a flock environment with 10 agents
    env = flock_env.SimpleFlockEnv(
        reward_func=flock_env.rewards.exponential_rewards, n_agents=10, i_range=0.1
    )
    env_params = env.default_params()

    # Reset the environment and get state and observation
    obs, state = env.reset(key_reset, env_params)
    # Sample random action for agents
    actions = jax.random.uniform(key_act, (10, 2))
    # Step the environment
    new_obs, new_state, rewards, dones = env.step(key_step, env_params, state, actions)
