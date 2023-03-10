# Flock Multi Agent RL Environment

Multi-agent RL environment based on Boids, implemented with
[JAX](https://github.com/google/jax)

![alt text](.github/images/rl_boids001.gif?raw=true)

The environment is based on popular [boids model](https://en.wikipedia.org/wiki/Boids)
where agents recreate flocking behaviours based on simple interaction rules.
The environment implements boids as a multi-agent reinforcement problem where each
boids take individual actions and have a unique localised view of the environment.

This environment has been built around the [gymnax](https://github.com/RobertTLange/gymnax)
API (a JAX version of the popular RL Gym API).

## Usage

See [`examples/ppo_example.ipynb`](/examples/ppo_example.ipynb) for an example
of training a Proximal-Policy-Optimisation based agent with this environment.

The package can and requirements can be installed using [poetry](https://python-poetry.org/docs/)
by running

```shell
poetry install
```

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
