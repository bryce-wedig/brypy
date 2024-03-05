import numpy as np
import pytest
# import numpy.testing as npt
# import unittest

import brypy.util as bp


# NB np.bool(False) is False evaluates to False
def test_check_negative_values():
    z = np.array([[-1, 0], [1, 2], [3, 4]])
    assert bp.check_negative_values(z) == True

    z = np.ones((4, 2))
    assert bp.check_negative_values(z) == False

    z = [np.array([-1, 0]), np.array([1, 2]), np.array([3, 4])]
    assert bp.check_negative_values(z) == True

    z = [np.ones((4, 2)), np.ones((4, 2))]
    assert bp.check_negative_values(z) == False


def test_replace_negatives_with_zeros():
    array = np.array([[-1, 0], [1, -2], [3, 4]])
    expected_result = np.array([[0, 0], [1, 0], [3, 4]])
    assert np.array_equal(bp.replace_negatives_with_zeros(array), expected_result)

    array = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[1, 2], [3, 4]])
    assert np.array_equal(bp.replace_negatives_with_zeros(array), expected_result)

    array = [np.array([-1, 0]), np.array([1, -2]), np.array([3, 4])]
    expected_result = [np.array([0, 0]), np.array([1, 0]), np.array([3, 4])]
    assert np.array_equal(bp.replace_negatives_with_zeros(array), expected_result)

    array = [np.ones((4, 2)), np.ones((4, 2))]
    expected_result = [np.ones((4, 2)), np.ones((4, 2))]
    assert np.array_equal(bp.replace_negatives_with_zeros(array), expected_result)


def test_percent_error():
    observed = 5
    exact = 4
    expected_result = 25.0
    assert bp.percent_error(observed, exact) == expected_result

    observed = 10
    exact = 10
    expected_result = 0.0
    assert bp.percent_error(observed, exact) == expected_result


def test_rotate_array():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    angle = 90
    expected_result = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
    assert np.array_equal(bp.rotate_array(array, angle), expected_result)

    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    angle = -45
    expected_result = np.array([[9, 6, 3], [8, 5, 2], [7, 4, 1]])
    assert np.array_equal(bp.rotate_array(array, angle), expected_result)

    array = np.array([[1, 2], [3, 4]])
    angle = 180
    expected_result = np.array([[4, 3], [2, 1]])
    assert np.array_equal(bp.rotate_array(array, angle), expected_result)

    array = np.array([[1]])
    angle = 270
    expected_result = np.array([[1]])
    assert np.array_equal(bp.rotate_array(array, angle), expected_result)


def test_get_indices_of_largest_values():
    np_array = np.array([1, 5, 3, 9, 2])
    num_points = 3
    expected_result = np.array([3, 1, 2])
    assert np.array_equal(bp.get_indices_of_largest_values(num_points, np_array), expected_result)

    np_array = np.array([1, 2, 3, 4, 5])
    num_points = 2
    expected_result = np.array([4, 3])
    assert np.array_equal(bp.get_indices_of_largest_values(num_points, np_array), expected_result)

    np_array = np.array([5, 4, 3, 2, 1])
    num_points = 4
    expected_result = np.array([0, 1, 2, 3])
    assert np.array_equal(bp.get_indices_of_largest_values(num_points, np_array), expected_result)

    np_array = np.array([1, 1, 1, 1, 1])
    num_points = 1
    expected_result = np.array([0])
    assert np.array_equal(bp.get_indices_of_largest_values(num_points, np_array), expected_result)


def test_get_indices_of_smallest_values():
    np_array = np.array([5, 2, 8, 1, 6])
    num_points = 3
    expected_result = np.array([3, 1, 0])
    assert np.array_equal(bp.get_indices_of_smallest_values(num_points, np_array), expected_result)

    np_array = np.array([1, 2, 3, 4, 5])
    num_points = 2
    expected_result = np.array([0, 1])
    assert np.array_equal(bp.get_indices_of_smallest_values(num_points, np_array), expected_result)

    np_array = np.array([5, 4, 3, 2, 1])
    num_points = 4
    expected_result = np.array([4, 3, 2, 1])
    assert np.array_equal(bp.get_indices_of_smallest_values(num_points, np_array), expected_result)

    np_array = np.array([1, 1, 1, 1, 1])
    num_points = 1
    expected_result = np.array([0])
    assert np.array_equal(bp.get_indices_of_smallest_values(num_points, np_array), expected_result)
    