import chex


@chex.dataclass
class EnvState:
    agent_positions: chex.Array
    agent_speeds: chex.Array
    agent_headings: chex.Array
    time: int


@chex.dataclass
class EnvParams:
    min_speed: float = 0.015
    max_speed: float = 0.025
    max_rotate: float = 0.025
    max_accelerate: float = 0.001
    square_min_range: float = 0.01**2
    collision_penalty: float = 0.1
