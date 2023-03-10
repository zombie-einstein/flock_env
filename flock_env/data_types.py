import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvState:
    agent_positions: jnp.array
    agent_speeds: jnp.array
    agent_headings: jnp.array
    time: int


@struct.dataclass
class EnvParams:
    min_speed: float = 0.015
    max_speed: float = 0.025
    max_rotate: float = 0.025
    max_accelerate: float = 0.001
    square_range: float = 0.1**2
    square_min_range: float = 0.01**2
