import datetime
import json
import os
import pickle as _pickle
import shutil
from csv import DictReader, DictWriter
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits


def percent_change(old, new):
    """
    Calculate the percent change between two values.

    Parameters
    ----------
    old : float
        The initial value.
    new : float
        The new value.

    Returns
    -------
    float
        The percent change from the old value to the new value.

    Examples
    --------
    >>> percent_change(50, 75)
    50.0
    >>> percent_change(100, 80)
    -20.0
    """
    return (new - old) / np.abs(old) * 100


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


def center_crop_image(array, shape):
    """
    Crop an array to the specified shape from the center.

    Parameters
    ----------
    array : numpy.ndarray
        The input 2D array to be cropped.
    shape : tuple of int
        The desired output shape as (height, width).

    Returns
    -------
    numpy.ndarray
        The center-cropped array with the specified shape.

    Examples
    --------
    >>> arr = np.ones((100, 100))
    >>> center_crop_image(arr, (50, 50)).shape
    (50, 50)
    """
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


def combine_all_csvs(path, filename):
    """
    Combine all CSV files in a directory into a single CSV file.

    Parameters
    ----------
    path : str
        The directory path containing CSV files to combine.
    filename : str
        The output filename for the combined CSV.

    Returns
    -------
    None
        Writes the combined CSV to the specified filename.
    """
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


def get_fits_data(fits_filepath, hdu_name):
    """
    Extract data from a specific HDU in a FITS file.

    Parameters
    ----------
    fits_filepath : str
        The path to the FITS file.
    hdu_name : str or int
        The name or index of the HDU to extract data from.

    Returns
    -------
    numpy.ndarray
        The data array from the specified HDU.
    """
    with fits.open(fits_filepath) as hdu_list:
        hdu_list.verify()
        data = hdu_list[hdu_name].data

    return data


def array_to_fits(array):
    """
    Write a numpy array to a FITS file named 'output.fits'.

    Parameters
    ----------
    array : numpy.ndarray
        The data array to write to the FITS file.

    Returns
    -------
    None
        Writes the array to 'output.fits' in the current directory.
    """
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=array))

    hdul.writeto('output.fits', overwrite=True)


def read_json(filepath):
    """
    Read and parse a JSON file.

    Parameters
    ----------
    filepath : str
        The path to the JSON file.

    Returns
    -------
    dict or list
        The parsed JSON content.
    """
    with open(filepath) as json_file:
        return json.load(json_file)


def batch_list(list, n):
    """
    Split a list into batches of size n.

    Parameters
    ----------
    list : list
        The list to be split into batches.
    n : int
        The size of each batch.

    Yields
    ------
    list
        Successive n-sized chunks from the input list.

    Examples
    --------
    >>> list(batch_list([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(list), n):
        yield list[i:i + n]


def combine_images(columns, space, images, filename):
    """
    Combine multiple images into a grid layout and save to a file.

    Parameters
    ----------
    columns : int
        The number of columns in the grid.
    space : int
        The spacing in pixels between images.
    images : list of str
        List of file paths to the images to combine.
    filename : str
        The output filename for the combined image.

    Returns
    -------
    None
        Saves the combined image to the specified filename.
    """
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
    """
    Unpickle all files in a directory matching a given prefix.

    Parameters
    ----------
    dir_path : str
        The directory path containing pickled files.
    prefix : str, optional
        Filter files by this prefix. Default is '' (all files).
    limit : int, optional
        Maximum number of files to unpickle. Default is None (no limit).

    Returns
    -------
    list
        A list of unpickled objects from the matching files, sorted alphabetically.
    """
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
    """
    Convert a number to a LaTeX-formatted scientific notation string.

    Parameters
    ----------
    input : float
        The number to convert to scientific notation.

    Returns
    -------
    str
        A LaTeX-formatted string in the form 'N\\cross10^{M}' where N is
        the coefficient rounded to 2 decimal places and M is the exponent.

    Examples
    --------
    >>> scientific_notation_string(1500)
    '1.5\\\\cross10^{3}'
    """
    # convert to Python scientific notion
    string = '{:e}'.format(input)
    num_string, exponent = string.split('e')
    num = str(round(float(num_string), 2))

    # handle exponent
    if exponent[0] == '+':
        _, power = exponent.split('+')
    elif exponent[0] == '-':
        _, power = exponent.split('-')
        power = '-' + power


    power = str(int(power))
    exponent = '10^{' + power + '}'

    return ''.join((num, '\cross', exponent))


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
