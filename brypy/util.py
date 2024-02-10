import datetime
import json
import os
import pickle as _pickle
import shutil
from collections import ChainMap
from csv import DictReader, DictWriter
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from omegaconf import OmegaConf


def percent_error(observed, exact):
    return (np.abs(observed - exact) / exact) * 100


def pad_rgb_array(rgb_array, pad, value):
    return np.stack([np.pad(rgb_array[:, :, i], (pad,), mode='constant', constant_values=value) for i in range(3)],
                    axis=2)


def rotate_array(array, angle):
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
    s = open(filepath, mode='r', encoding='utf-8-sig').read()
    open(filepath, mode='w', encoding='utf-8').write(s)


def dict_list_to_csv(dict_list, csv_filepath):
    if dict_list is not None:
        keys = get_dict_keys_as_list(dict_list[0])

        with open(csv_filepath, 'w') as csv_file:
            writer = DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dict_list)
    else:
        raise Exception('Dictionary list is empty')


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


def center_crop_image(array, shape):
    y_out, x_out = shape
    tuple = array.shape
    y, x = tuple[0], tuple[1]
    x_start = (x // 2) - (x_out // 2)
    y_start = (y // 2) - (y_out // 2)
    return array[y_start:y_start + y_out, x_start:x_start + x_out]


def get_indices_of_largest_values(num_points, np_array):
    """
    Return indices of largest values of np.array where first element is index of largest value
    """
    indices_of_sorted = np.argsort(np_array)
    return np.flip(indices_of_sorted[-num_points:])


def get_indices_of_smallest_values(num_points, np_array):
    """
    Return indices of smallest values of np.array where first element is index of smallest value
    """
    indices_of_sorted = np.argsort(np_array)
    return indices_of_sorted[:num_points]


def print_execution_time(start, stop):
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def pickle(path, thing):
    with open(path, 'ab') as results_file:
        _pickle.dump(thing, results_file)


def unpickle(path):
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
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(path):
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
    if os.path.exists(path):
        os.remove(path)


def get_dict_keys_as_list(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list


def get_today_str():
    return str(datetime.date.today())
