import typing
from math import floor

import chex
import esquilax
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from . import data_types, steps, utils


class BasePredatorPreyEnv(
    esquilax.ml.rl.Environment[
        data_types.PredatorPreyState, data_types.PredatorPreyParams
    ]
):
    def __init__(
        self,
        *,
        n_predators: int,
        n_prey: int,
        prey_vision_range: float,
        predator_vision_range: float,
        n_vision: int,
        agent_radius: float,
        sparse_rewards: bool,
    ):
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.predator_vision_range = predator_vision_range
        self.n_predator_view_bins = floor(1.0 / predator_vision_range)
        self.prey_vision_range = prey_vision_range
        self.n_prey_vision_bins = floor(1.0 / prey_vision_range)
        self.n_vision = n_vision
        self.agent_radius = agent_radius
        self.sparse_rewards = sparse_rewards

    def default_params(self) -> data_types.PredatorPreyParams:
        return data_types.PredatorPreyParams()

    def reset(
        self, key: chex.PRNGKey, params: data_types.PredatorPreyParams
    ) -> typing.Tuple[data_types.PredatorPrey, data_types.PredatorPreyState]:
        k_predator, k_prey = jax.random.split(key)
        predators = utils.init_boids(
            self.n_predators, params.predator_params, k_predator
        )
        prey = utils.init_boids(self.n_prey, params.prey_params, k_prey)
        new_state = data_types.PredatorPreyState(prey=prey, predators=predators, step=0)
        obs = self.get_obs(key, params, new_state)
        return jax.lax.stop_gradient((obs, new_state))

    def step(
        self,
        key: chex.PRNGKey,
        params: data_types.PredatorPreyParams,
        state: data_types.PredatorPreyState,
        actions: data_types.PredatorPrey,
    ) -> typing.Tuple[
        data_types.PredatorPrey,
        data_types.PredatorPreyState,
        data_types.PredatorPrey,
        data_types.PredatorPrey,
    ]:
        predators = utils.update_state(
            key, params.predator_params, state.predators, actions.predator
        )
        prey = utils.update_state(key, params.prey_params, state.prey, actions.prey)

        state = data_types.PredatorPreyState(
            predators=predators,
            prey=prey,
            step=state.step + 1,
        )

        if self.sparse_rewards:
            prey_rewards = esquilax.transforms.spatial(
                steps.sparse_prey_rewards,
                reduction=jnp.add,
                default=0.0,
                include_self=False,
                n_bins=floor(0.5 / self.agent_radius),
                i_range=2 * self.agent_radius,
            )(
                key,
                params.prey_penalty,
                None,
                None,
                pos=prey.position,
                pos_b=predators.position,
            )
            predator_rewards = esquilax.transforms.nearest_neighbour(
                steps.sparse_predator_rewards,
                default=0.0,
                n_bins=floor(0.5 / self.agent_radius),
                i_range=2 * self.agent_radius,
            )(
                key,
                params.predator_reward,
                None,
                None,
                pos=predators.position,
                pos_b=prey.position,
            )
        else:
            prey_rewards = esquilax.transforms.spatial(
                steps.distance_prey_rewards,
                reduction=jnp.add,
                default=0.0,
                include_self=False,
                n_bins=self.n_prey_vision_bins,
                i_range=self.prey_vision_range,
            )(
                key,
                params.prey_penalty,
                prey,
                predators,
                pos=prey.position,
                pos_b=predators.position,
                i_range=self.prey_vision_range,
            )
            predator_rewards = esquilax.transforms.spatial(
                steps.distance_predator_rewards,
                reduction=jnp.add,
                default=0.0,
                include_self=False,
                n_bins=self.n_predator_view_bins,
                i_range=self.predator_vision_range,
            )(
                key,
                params.predator_reward,
                predators,
                prey,
                pos=predators.position,
                pos_b=prey.position,
                i_range=self.prey_vision_range,
            )

        rewards = data_types.PredatorPrey(
            predator=predator_rewards,
            prey=prey_rewards,
        )
        obs = self.get_obs(key, params, state)
        done = self.is_terminal(params, state)
        return jax.lax.stop_gradient((obs, state, rewards, done))

    def get_obs(
        self,
        key: chex.PRNGKey,
        params: data_types.PredatorPreyParams,
        state: data_types.PredatorPreyState,
    ) -> data_types.PredatorPrey:
        prey_obs_predators = esquilax.transforms.spatial(
            steps.view,
            reduction=jnp.minimum,
            default=jnp.ones((self.n_vision,)),
            include_self=False,
            n_bins=self.n_prey_vision_bins,
            i_range=self.prey_vision_range,
        )(
            key,
            (params.prey_params.view_angle, self.agent_radius),
            state.prey,
            state.predators,
            pos=state.prey.position,
            pos_b=state.predators.position,
            n_view=self.n_vision,
            i_range=self.prey_vision_range,
        )
        prey_obs_prey = esquilax.transforms.spatial(
            steps.view,
            reduction=jnp.minimum,
            default=jnp.ones((self.n_vision,)),
            include_self=False,
            n_bins=self.n_prey_vision_bins,
            i_range=self.prey_vision_range,
        )(
            key,
            (params.predator_params.view_angle, self.agent_radius),
            state.prey,
            state.prey,
            pos=state.prey.position,
            n_view=self.n_vision,
            i_range=self.prey_vision_range,
        )
        predator_obs_prey = esquilax.transforms.spatial(
            steps.view,
            reduction=jnp.minimum,
            default=jnp.ones((self.n_vision,)),
            include_self=False,
            n_bins=self.n_predator_view_bins,
            i_range=self.predator_vision_range,
        )(
            key,
            (params.predator_params.view_angle, self.agent_radius),
            state.predators,
            state.prey,
            pos=state.predators.position,
            pos_b=state.prey.position,
            n_view=self.n_vision,
            i_range=self.predator_vision_range,
        )
        predator_obs_predator = esquilax.transforms.spatial(
            steps.view,
            reduction=jnp.minimum,
            default=jnp.ones((self.n_vision,)),
            include_self=False,
            n_bins=self.n_predator_view_bins,
            i_range=self.predator_vision_range,
        )(
            key,
            (params.predator_params.view_angle, self.agent_radius),
            state.predators,
            state.predators,
            pos=state.predators.position,
            n_view=self.n_vision,
            i_range=self.predator_vision_range,
        )

        predator_obs = jnp.hstack([predator_obs_prey, predator_obs_predator])
        prey_obs = jnp.hstack([prey_obs_predators, prey_obs_prey])

        return data_types.PredatorPrey(
            predator=predator_obs,
            prey=prey_obs,
        )

    def is_terminal(
        self, params: data_types.PredatorPreyParams, state: data_types.PredatorPreyState
    ) -> data_types.PredatorPrey:
        return data_types.PredatorPrey(
            predator=jnp.full((self.n_predators,), False),
            prey=jnp.full((self.n_prey,), False),
        )

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        Action space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            Action space description
        """
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        Observation space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            Observation space description
        """
        return spaces.Box(0, 1.0, shape=(2 * self.n_vision,), dtype=jnp.float32)

    def state_space(self, params: data_types.EnvParams) -> spaces.Space:
        """
        State space description

        Parameters
        ----------
        params: flock_env.data_types.EnvParams
            Environment parameters.

        Returns
        -------
        gymnax.spaces.Space
            State space description.
        """
        return spaces.Dict(
            dict(
                agent_positions=spaces.Box(0.0, 1.0, (2,), jnp.float32),
                agent_speeds=spaces.Box(
                    params.min_speed,
                    params.max_speed,
                    (),
                    jnp.float32,
                ),
                agent_headings=spaces.Box(
                    0.0,
                    2 * jnp.pi,
                    (),
                    jnp.float32,
                ),
                time=spaces.Discrete(jnp.finfo(jnp.int32).max),
            )
        )
