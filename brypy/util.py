import datetime
import json
import os
import pickle as _pickle
import shutil
# import h5py
from collections import ChainMap
from csv import DictReader, DictWriter
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from omegaconf import OmegaConf


def percent_change(a, b):
    """
    Calculate the percentage change between two values.

    Parameters
    ----------
    a : float
        The first value.
    b : float
        The second value.

    Returns
    -------
    float
        The percentage change between the two values.

    Examples
    --------
    >>> percent_change(10, 20)
    100.0

    >>> percent_change(10, 15)
    50.0
    """
    return np.abs(a - b) / a * 100


def percent_difference(a, b):
    """
    Calculate the percentage difference between two values.

    Parameters
    ----------
    a : float
        The first value.
    b : float
        The second value.

    Returns
    -------
    float
        The percentage difference between the two values.

    Examples
    --------
    >>> percent_difference(10, 20)
    100.0

    >>> percent_difference(10, 15)
    50.0
    """
    return np.abs(a - b) / ((a + b) / 2) * 100


# def h5_tree(val, pre=''):
#     """
#     Print the tree structure of an HDF5 file.
#     https://stackoverflow.com/questions/61133916/is-there-in-python-a-single-function-that-shows-the-full-structure-of-a-hdf5-fi

#     Parameters
#     ----------
#     val
#     pre

#     Returns
#     -------

#     """
#     items = len(val)
#     for key, val in val.items():
#         items -= 1
#         if items == 0:
#             # the last item
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '└── ' + key)
#                 h5_tree(val, pre+'    ')
#             else:
#                 print(pre + '└── ' + key + ' (%d)' % len(val))
#         else:
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '├── ' + key)
#                 h5_tree(val, pre+'│   ')
#             else:
#                 print(pre + '├── ' + key + ' (%d)' % len(val))


def check_negative_values(array):
    """
    Check if there are any negative values in the given array.

    Parameters
    ----------
    array : array_like
        The input array to check for negative values.

    Returns
    -------
    bool
        Returns True if there are negative values in the array, otherwise False.

    Notes
    -----
    This function supports both 1-dimensional and multi-dimensional arrays.

    Examples
    --------
    >>> check_negative_values([1, 2, 3])
    False

    >>> check_negative_values([-1, 0, 1])
    True

    >>> check_negative_values(np.array([[1, 2], [-3, 4]]))
    True
    """
    if isinstance(array, list):
        for a in array:
            if np.any(a < 0):
                return True
            else:
                return False
    else:
        if np.any(array < 0):
            return True
        else:
            return False


def replace_negatives_with_zeros(array):
    """
    Replace negative values in the input array with zeros.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Array with negative values replaced by zeros.
    """
    return np.where(array < 0, 0, array)


def resize_with_pixels_centered(array, oversample_factor):
    """
    Resize the input array with centered pixels using the specified oversample factor.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be resized. It must be square.
    oversample_factor : int
        The factor by which to oversample the array. It must be odd.

    Returns
    -------
    numpy.ndarray
        The resized array with centered pixels.

    Raises
    ------
    Exception: If the oversample factor is even.
    Exception: If the input array is not square.
    """

    if oversample_factor % 2 == 0:
        raise Exception('Oversampling factor must be odd')
    
    x, y = array.shape
    if x != y:
        raise Exception('Array must be square')
    
    flattened_array = array.flatten()
    oversample_grid = np.zeros((x * oversample_factor, x * oversample_factor))

    k = 0
    for i, row in enumerate(oversample_grid):
        for j, _ in enumerate(row):
            if not (i % oversample_factor) - ((oversample_factor - 1) / 2) == 0:
                continue
            if (j % oversample_factor) - ((oversample_factor - 1) / 2) == 0:
                oversample_grid[i][j] = flattened_array[k]
                k += 1

    return oversample_grid


def center_crop_image(array, shape):
    if array.shape == shape:
        return array

    y_out, x_out = shape
    tuple = array.shape
    y, x = tuple[0], tuple[1]
    x_start = (x // 2) - (x_out // 2)
    y_start = (y // 2) - (y_out // 2)
    return array[y_start:y_start + y_out, x_start:x_start + x_out]


def percent_error(observed, exact):
    """
    Calculate the percent error between an observed value and an exact value.

    Parameters
    ----------
    observed : float
        The observed value.
    exact : float
        The exact value.

    Returns
    -------
    float
        The percent error between the observed and exact values.

    Examples
    --------
    >>> percent_error(5, 4)
    25.0

    >>> percent_error(10, 10)
    0.0
    """
    return (np.abs(observed - exact) / exact) * 100


def pad_rgb_array(rgb_array, pad, value):
    return np.stack([np.pad(rgb_array[:, :, i], (pad,), mode='constant', constant_values=value) for i in range(3)],
                    axis=2)


def rotate_array(array, angle):
    """
    Rotate a 2D numpy array by a given angle.

    Parameters
    ----------
    array : numpy.ndarray 
        The input array to be rotated.
    angle : float
        The angle of rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The rotated array.

    """
    pil_image = Image.fromarray(array)
    rotated_pil_image = pil_image.rotate(angle)
    return np.asarray(rotated_pil_image)


def hydra_to_dict(config):
    container = OmegaConf.to_container(config, resolve=True)
    return dict(ChainMap(*container))


def combine_all_csvs(path, filename):
    # list all files in directory
    csv_files = [f for f in os.listdir(path) if not f.startswith('.')]

    # concatenate CSVs
    pd_list = []

    for f in csv_files:
        pd_list.append(pd.read_csv(os.path.join(path, f)))

    df_res = pd.concat(pd_list, ignore_index=True)

    df_res.to_csv(filename)


def remove_bom(filepath):
    """
    Remove the byte-order mark (BOM) from a CSV file

    Parameters
    ----------
    filepath : str
        Path to the CSV file
    """
    s = open(filepath, mode='r', encoding='utf-8-sig').read()
    open(filepath, mode='w', encoding='utf-8').write(s)


# TODO fix
# def dict_list_to_csv(dict_list, csv_filepath):
#     if dict_list is not None:
#         keys = get_dict_keys_as_list(dict_list[0])

#         with open(csv_filepath, 'w') as csv_file:
#             writer = DictWriter(csv_file, fieldnames=keys)
#             writer.writeheader()
#             writer.writerows(dict_list)
#     else:
#         raise Exception('Dictionary list is empty')


def csv_to_dict_list(csv_filepath):
    with open(csv_filepath, mode='r', encoding='utf-8-sig') as f:
        dict_reader = DictReader(f)
        list_to_return = list(dict_reader)

    return list_to_return


def get_fits_data(fits_filepath, hdu_name):
    with fits.open(fits_filepath) as hdu_list:
        hdu_list.verify()
        data = hdu_list[hdu_name].data

    return data


def array_to_fits(array):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=array))

    hdul.writeto('output.fits', overwrite=True)


def read_json(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def combine_images(columns, space, images, filename):
    # calculate number of rows based on columns
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1

    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max * columns + (space * columns) - space
    background_height = height_max * rows + (space * rows) - space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0

    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max - img.width) / 2)
        y_offset = int((height_max - img.height) / 2)
        background.paste(img, (x + x_offset, y + y_offset))
        x += width_max + space

        if (i + 1) % columns == 0:
            y += height_max + space
            x = 0

    background.save(filename)


def get_indices_of_largest_values(num_points, np_array):
    """
    Returns the indices of the largest values in the given NumPy array.

    Parameters
    ----------
    num_points : int
        The number of indices to return.
    np_array : numpy.ndarray
        The input array.

    Returns
    -------
    numpy.ndarray
        An array of indices corresponding to the largest values in the input array.

    Examples
    --------
    >>> arr = np.array([1, 5, 3, 9, 2])
    >>> get_indices_of_largest_values(3, arr)
    array([3, 1, 2])

    >>> arr = np.array([10, 20, 30, 40, 50])
    >>> get_indices_of_largest_values(2, arr)
    array([4, 3])
    """
    indices_of_sorted = np.argsort(np_array)
    return np.flip(indices_of_sorted[-num_points:])


def get_indices_of_smallest_values(num_points, np_array):
    """
    Return the indices of the smallest values in the given Numpy array.

    Parameters
    ----------
    num_points : int
        The number of indices to return.
    np_array : numpy.ndarray
        The input array.

    Returns
    -------
    numpy.ndarray
        An array of indices corresponding to the smallest values in the input array.
    """
    indices_of_sorted = np.argsort(np_array)
    return indices_of_sorted[:num_points]


def print_execution_time(start, stop):
    """
    Print the execution time between two given timestamps.

    Parameters
    ----------
    start : float
        The start timestamp.
    stop : float
        The stop timestamp.

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> start = time.time()
    >>> # Some code to measure execution time
    >>> stop = time.time()
    >>> print_execution_time(start, stop)
    Execution time: 0:00:05

    """
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def pickle(path, thing):
    """
    Pickle an object and save it to a file. Note that the file will be overwritten if it already exists.

    Parameters
    ----------
    path : str 
        The path to the file where the object will be saved.
    thing : object
        The object to be pickled and saved.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If the user does not have permission to write to the specified file.

    """
    with open(path, 'wb') as results_file:
        _pickle.dump(thing, results_file)


def unpickle(path):
    """
    Unpickles an object from a file.

    Parameters
    ----------
    path : str
        The path to the file containing the pickled object.

    Returns
    -------
    object
        The unpickled object.

    Raises
    ------
    FileNotFoundError
        If the file specified by `path` does not exist.
    EOFError
        If the end of the file is reached unexpectedly.
    _pickle.UnpicklingError
        If the pickled object cannot be unpickled.
    """
    with open(path, 'rb') as results_file:
        result = _pickle.load(results_file)
    return result


def unpickle_all(dir_path, prefix='', limit=None):
    file_list = glob(dir_path + f'/{prefix}*')
    sorted_list = sorted(file_list)
    if limit is not None:
        return [unpickle(i) for i in sorted_list[:limit] if os.path.isfile(i)]
    else:
        return [unpickle(i) for i in sorted_list if os.path.isfile(i)]


def create_directory_if_not_exists(path):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        The path of the directory to be created.

    Returns
    -------
    None

    """
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(path):
    """
    Clear all files and directories within the specified path.

    Parameters
    ----------
    path : str
        The path to the directory to be cleared.

    Returns
    -------
    None

    """
    for i in glob(path + '/*'):
        if os.path.isfile(i):
            os.remove(i)
        else:
            shutil.rmtree(i)


def scientific_notation_string(input):
    return '{:.2e}'.format(input)


# TODO finish
# def scientific_notation_string(input):
#     # convert to Python scientific notion
#     string = '{:e}'.format(input)
#     num_string, exponent = string.split('e')
#     num = str(round(float(num_string), 2))

#     # handle exponent
#     if exponent[0] == '+':
#         _, power = exponent.split('+')
#     elif exponent[0] == '-':
#         _, power = exponent.split('-')
#         power = '-' + power


#     power = str(int(power))
#     exponent = '10^{' + power + '}'

#     return ''.join((num, '\cross', exponent))


def delete_if_exists(path):
    """
    Delete a file if it exists.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        os.remove(path)


def get_today_str():
    """
    Get the current date as a string.

    Returns
    -------
    str
        The current date in the format 'YYYY-MM-DD'.
    """
    return str(datetime.date.today())
