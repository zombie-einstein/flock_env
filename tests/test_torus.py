import pytest

from flock_env import (
    _product_difference,
    _torus_vectors,
    _shortest_vec,
    _distances,
    _relative_angles,
    _relative_headings
)
import numpy as np


def test_product_difference():
    x = np.arange(5).astype(np.float32)
    y = _product_difference(x, 3)
    assert y.shape == (3, 4)
    assert np.array_equal(y[0], [1, 2, 3, 4])
    assert np.array_equal(y[1], [-1, 1, 2, 3])
    assert np.array_equal(y[2], [-2, -1, 1, 2])


test_data = [
    (np.array([1, 2]).astype(np.float32), 3.0, 2),
    (np.array([1, 9]).astype(np.float32), 10.0, 2),
    (np.array([1, 2]).astype(np.float32), 10.0, 2),
    (np.array([9, 1]).astype(np.float32), 10.0, 2),
    (np.array([0.1, 0.5]).astype(np.float32), 1.0, 2)
]


@pytest.mark.parametrize("x, l, n", test_data)
def test_torus_vectors(x, l, n):
    y1, y2 = _torus_vectors(x, l, n)

    assert y1.shape == (2, 1)
    assert y2.shape == (2, 1)

    assert np.isclose((x[0] + y1[0][0]) % l, x[1])
    assert np.isclose((x[0] + y2[0][0]) % l, x[1])
    assert np.isclose((x[1] + y1[1][0]) % l, x[0])
    assert np.isclose((x[1] + y2[1][0]) % l, x[0])


@pytest.mark.parametrize("x, l, n", test_data)
def test_shortest_torus_vectors(x, l, n):
    y1, y2 = _torus_vectors(x, l, n)
    y = _shortest_vec(x, l, n)

    assert y.shape == (2, 1)

    assert np.abs(y[0][0]) == min(np.abs(y1[0][0]), np.abs(y2[0][0]))
    assert np.abs(y[1][0]) == min(np.abs(y1[1][0]), np.abs(y2[1][0]))


def test_distances():
    x = _distances(np.array([[1], [1]]).astype(np.float32),
                   np.array([[-1], [2]]).astype(np.float32))
    assert x.shape == (2, 1)
    assert np.isclose(x[0][0], np.sqrt(2))
    assert np.isclose(x[1][0], np.sqrt(5))


def test_relative_angles():
    xs = np.array([[0], [1], [1]]).astype(np.float32)
    ys = np.array([[1], [0], [1]]).astype(np.float32)
    ts = np.array([np.pi,
                   1.5*np.pi,
                   0.5*np.pi]).astype(np.float32)

    r = _relative_angles(xs, ys, ts)

    assert r.shape == (3, 1)
    assert np.isclose(r[0][0], 0.5)
    assert np.isclose(r[1][0], -0.5)
    assert np.isclose(r[2][0], 0.75)


def test_relative_headings():
    t = np.array([0, np.pi/2, np.pi]).astype(np.float32)
    r = _relative_headings(t)

    assert r.shape == (3, 2)
    assert np.allclose(r[0], np.array([0.5, -1.0]))
    assert np.allclose(r[1], np.array([-0.5, 0.5]))
