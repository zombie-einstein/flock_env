import jax.numpy as jnp
import pytest

from flock_env import utils


@pytest.mark.parametrize(
    "vec_a, vec_b, expected",
    [
        ([0.8, 0.5], [0.9, 0.5], [0.1, 0.0]),
        ([0.8, 0.5], [0.1, 0.5], [0.3, 0.0]),
        ([0.8, 0.5], [0.2, 0.5], [0.4, 0.0]),
        ([0.8, 0.5], [0.4, 0.5], [-0.4, 0.0]),
        ([0.8, 0.5], [0.6, 0.5], [-0.2, 0.0]),
        ([0.2, 0.5], [0.8, 0.5], [-0.4, 0.0]),
        ([0.5, 0.8], [0.5, 0.2], [0.0, 0.4]),
        ([0.5, 0.2], [0.5, 0.8], [0.0, -0.4]),
    ],
)
def test_shortest_vector(vec_a, vec_b, expected):

    shortest_vec = utils.shortest_vector(jnp.array(vec_a), jnp.array(vec_b), 1.0)

    assert jnp.allclose(jnp.array(expected), shortest_vec)


@pytest.mark.parametrize(
    "vec_a, vec_b, expected",
    [
        ([0.8, 0.5], [0.8, 0.5], 0),
        ([0.8, 0.5], [0.5, 0.5], 0.3),
        ([0.8, 0.5], [0.1, 0.5], 0.3),
        ([0.1, 0.5], [0.8, 0.5], 0.3),
        ([0.5, 0.8], [0.5, 0.5], 0.3),
        ([0.5, 0.8], [0.5, 0.1], 0.3),
        ([0.5, 0.1], [0.5, 0.8], 0.3),
    ],
)
def test_shortest_distance(vec_a, vec_b, expected):

    d = utils.shortest_distance(jnp.array(vec_a), jnp.array(vec_b), 1.0, norm=True)

    assert jnp.isclose(d, expected)
