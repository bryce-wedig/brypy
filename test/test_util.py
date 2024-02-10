import numpy as np
import pytest
# import numpy.testing as npt
# import unittest

from brypy import util as bp


def test_check_negative_values():
    z = np.array([-1, 0], [1, 2], [3, 4])
    assert bp.check_negative_values(z) is True

    z = np.ones((4, 2))
    assert bp.check_negative_values(z) is False


def test_replace_negatives_with_zeros():
    # TODO 


# def test_resize_with_pixels_centered():
    # TODO check oversample factor even exception

    # TODO check non-square array

    # TODO test happy path


# def test_center_crop_image():
    # TODO test odd side

    # TODO test even side

    # TODO test uneven sides

# TODO test largest values method too

# def test_get_indices_of_smallest_values():
#     z = np.array([])
#     indices = 
#     assert indices 

#     z = np.array([], [])
    