# Boid Multi-Agent RL Environment & Multi-Agent RL Agent

![alt text](images/009_161.gif?raw=true)![alt text](images/022_081.gif?raw=true)

## Introduction

This repo contains modules for:

- A multi-agent RL training environment based on the boid flocking
model consisting of multiple independent agents (boids) navigating
the environment alongside other members of the flock. The aim
being to reproduce the emergent flocking behaviour seen in nature.
- An experience buffer designed to allow a single RL agent to 
learn a policy for multiple homogenous agents simultaneously (along with a 
modified training loop) to be able to generate emergent behaviours.

See `usage.ipynb` for examples of usage of the environment and
the RL agent. Examples using a DQN have been included, though the environment
and design of the buffer should be reasonably general.

## Requirements

Requirements to use the environment and examples can be found in 
`requirements.txt`. The model makes use of [numba](https://numba.pydata.org/)
for performance. 

## Flock Environment

See `flock_env.py`. A RL training environment designed to comply
with the Open-AI Gym API. The environment itself is based on the 
[boid model](https://en.wikipedia.org/wiki/Boids) where the flock
is formed from a set of independent agents (termed boids) who's aim
is generally to form a close flock whilst avoiding collisions
alongside possible additional desired (environmental obstacles etc.)
by altering their trajectory based on the current state of the flock.

The environment treats each agent separately, so requires that the 
actions provided by the RL agent(s) at each step are an 
array of actions for each agent in the flock, in return the 
returned observations and rewards also contain values for each
agent. The observations are local views for each agent, i.e. the 
locations of other boids relative to that particular agent.

Currently the implemented environment `DiscreteActionFlock` is based 
on a discrete action space where agents 'steer' by fixed amounts 
at each step and maintain a fixed speed.

The aim is to add a continuous action space version in future 
iterations alongside additional environmental features.

## Multi-Agent Experience Buffer

See `agent_experience_buffer.py`. This buffer is designed to allow 
a single RL agent to learn from experience gathered from multiple 
homogenous agents. The buffer is implemented as 2d arrays in the shape
`[step][agent]`. At each step the values returned from the multi-agent
environment are inserted for each agent. 

Sampling the buffer is done in the same manner as normal RL agents, 
with sample taken uniformly from the step and agent dimensions.
