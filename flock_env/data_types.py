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
    min_speed: float = 0.01
    max_speed: float = 0.05
    max_rotate: float = 0.1
    max_accelerate: float = 0.025
    square_range: float = 0.01
    square_min_range: float = 0.0001
