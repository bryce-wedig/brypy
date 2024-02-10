import numpy as np
import pytest
# import numpy.testing as npt
# import unittest

import brypy.util as bp


def test_check_negative_values():
    # TODO test if list of arrays

    z = np.array([[-1, 0], [1, 2], [3, 4]])
    assert bp.check_negative_values(z) == True
    # NB np.bool(False) is False evaluates to False

    z = np.ones((4, 2))
    assert bp.check_negative_values(z) == False


# def test_replace_negatives_with_zeros():
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
    