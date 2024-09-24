import chex


@chex.dataclass
class Boid:
    position: chex.Array
    speed: float
    heading: float


@chex.dataclass
class EnvState:
    boids: Boid
    step: int


@chex.dataclass
class EnvParams:
    min_speed: float = 0.015
    max_speed: float = 0.025
    max_rotate: float = 0.025
    max_accelerate: float = 0.001
    collision_penalty: float = 0.1
    agent_radius: float = 0.01
    view_angle: float = 0.5
