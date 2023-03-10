# Flock Multi Agent RL Environment

Multi-agent RL environment based on Boids, implemented with
[JAX](https://github.com/google/jax)

![alt text](.github/images/rl_boids001.gif?raw=true)

The environment is based on popular [boids model](https://en.wikipedia.org/wiki/Boids)
where agents recreate flocking behaviours based on simple interaction rules.
The environment implements boids as a multi-agent reinforcement problem where each
boid takes individual actions and have a individual localised view of the environment.

This environment has been built around the [gymnax](https://github.com/RobertTLange/gymnax)
API (a JAX version of the popular RL Gym API):

```python
import flock_env
import jax

key = jax.random.PRNGKey(101)
key_reset, key_act, key_step = jax.random.split(key)

# Initialise a flock environment with 10 agents
env = flock_env.SimpleFlockEnv(
    reward_func=flock_env.rewards.exponential_rewards,
    n_agents=10
)
env_params = env.default_params

# Reset the environment and get state and observation
obs, state = env.reset(key_reset, env_params)
# Sample random action for agents
actions = jax.random.uniform(key_act, (10, 2))
# Step the environment
new_obs, new_state, rewards, dones, _ = env.step_env(
    key_step, state, actions, env_params
)
```

## Usage

See [`examples/ppo_example.ipynb`](/examples/ppo_example.ipynb) for an example
of training a Proximal-Policy-Optimisation based agent with this environment
(using my [JAX implementation of PPO](https://github.com/zombie-einstein/JAX-PPO)).

The package can and requirements can be installed using [poetry](https://python-poetry.org/docs/)
by running

```shell
poetry install
```

> :warning: Generating observations currently compares all pairs of agents
> so performance scales as $n^2$ with the number of agents. This means performance
> may not be great past hundreds of agents.

## TODO

- More complex observation spaces, e.g. ray-casting view model
- Objects/obstacles in the environment
- More efficient agent observation generation

## Previous Version

The previous version of this project built around Numba can be found in
[`/deprecated`](/deprecated)

## Developers

### Pre-Commit Hooks

Pre commit hooks can be installed by running

```bash
pre-commit install
```

Pre-commit checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```
