import jax
import jax_ppo
from esquilax.ml import rl

import flock_env
from examples import ppo_agent


def test_agent():
    key = jax.random.PRNGKey(101)

    n_agents = 5
    n_actions = 3
    n_epochs = 4
    n_steps = 7
    n_env = 2

    env = flock_env.SimpleFlockEnv(
        reward_func=flock_env.rewards.exponential_rewards,
        n_agents=n_agents,
        i_range=0.1,
    )
    env_params = env.default_params()
    obs_shape = env.observation_space(env_params).shape

    agent = ppo_agent.PPOAgent(
        jax_ppo.default_params,
        1,
        8,
        4,
    )
    agent_state = agent.init_state(
        key,
        obs_shape,
        n_actions,
        0.1,
        layer_width=4,
        n_layers=1,
    )

    observations = jax.numpy.zeros((n_agents,) + obs_shape)

    actions, values = agent.sample_actions(key, agent_state, observations)

    assert actions.shape == (n_agents, n_actions)
    assert isinstance(values, dict)
    assert sorted(values.keys()) == ["log_likelihood", "value"]
    assert values["log_likelihood"].shape == (n_agents,)
    assert values["value"].shape == (n_agents,)

    actions, values = agent.sample_actions(key, agent_state, observations, greedy=True)

    assert actions.shape == (n_agents, n_actions)
    assert isinstance(values, dict)
    assert sorted(values.keys()) == ["log_likelihood", "value"]
    assert values["log_likelihood"].shape == (n_agents,)
    assert values["value"].shape == (n_agents,)

    updated_agent, rewards, losses = rl.train(
        key,
        agent,
        agent_state,
        env,
        env_params,
        n_epochs,
        n_env,
        n_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, rl.AgentState)
    assert rewards.shape == (n_epochs, n_env, n_steps, n_agents)
    assert isinstance(losses, dict)
