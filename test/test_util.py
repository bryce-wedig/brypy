import datetime
import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import brypy.util as bp


def test_percent_change():
    assert bp.percent_change(50, 75) == 50.0
    assert bp.percent_change(100, 80) == -20.0
    assert bp.percent_change(100, 100) == 0.0
    assert bp.percent_change(-50, -25) == 50.0


def test_percent_difference():
    assert bp.percent_difference(10, 20) == pytest.approx(66.666666, rel=1e-4)
    assert bp.percent_difference(10, 10) == 0.0
    assert bp.percent_difference(100, 200) == pytest.approx(66.666666, rel=1e-4)


# NB np.bool(False) is False evaluates to False
def test_check_negative_values():
    z = np.array([[-1, 0], [1, 2], [3, 4]])
    assert bp.check_negative_values(z)

    z = np.ones((4, 2))
    assert not bp.check_negative_values(z)

    z = [np.array([-1, 0]), np.array([1, 2]), np.array([3, 4])]
    assert bp.check_negative_values(z)

    z = [np.ones((4, 2)), np.ones((4, 2))]
    assert not bp.check_negative_values(z)


def test_replace_negatives_with_zeros():
    array = np.array([[-1, 0], [1, -2], [3, 4]])
    expected_result = np.array([[0, 0], [1, 0], [3, 4]])
    assert np.array_equal(bp.replace_negatives_with_zeros(array), expected_result)

    array = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[1, 2], [3, 4]])
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

    # Test warning is issued when all values are identical
    np_array = np.array([1, 1, 1, 1, 1])
    num_points = 2  # Request 2 points so duplicates are in the result
    expected_result = np.array([4, 3])  # Later indices first due to flip
    with pytest.warns(UserWarning, match="identical"):
        assert np.array_equal(bp.get_indices_of_largest_values(num_points, np_array), expected_result)

    # Test warning is issued when some (but not all) values are identical
    np_array = np.array([1, 5, 5, 2, 3])
    num_points = 3
    expected_result = np.array([2, 1, 4])  # indices of the two 5s and the 3, later indices first
    with pytest.warns(UserWarning, match="identical"):
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


def test_center_crop_image():
    # Test basic cropping
    arr = np.arange(100).reshape(10, 10)
    result = bp.center_crop_image(arr, (4, 4))
    assert result.shape == (4, 4)

    # Test that center values are preserved
    arr = np.zeros((10, 10))
    arr[4:6, 4:6] = 1  # Set center to 1
    result = bp.center_crop_image(arr, (2, 2))
    assert np.all(result == 1)

    # Test same shape returns same array
    arr = np.ones((5, 5))
    result = bp.center_crop_image(arr, (5, 5))
    assert np.array_equal(result, arr)


def test_rotate_array():
    # Test 180 degree rotation
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    result = bp.rotate_array(arr, 180)
    expected = np.array([[4, 3], [2, 1]])
    assert np.array_equal(result, expected)

    # Test 0 degree rotation (no change)
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    result = bp.rotate_array(arr, 0)
    assert np.array_equal(result, arr)


def test_combine_all_csvs():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV files
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        df1.to_csv(os.path.join(tmpdir, 'file1.csv'), index=False)
        df2.to_csv(os.path.join(tmpdir, 'file2.csv'), index=False)

        output_file = os.path.join(tmpdir, 'combined.csv')
        bp.combine_all_csvs(tmpdir, output_file)

        result = pd.read_csv(output_file)
        assert len(result) == 4
        assert list(result['a']) == [1, 2, 5, 6] or list(result['a']) == [5, 6, 1, 2]


def test_remove_bom():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        # Write file with BOM
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            f.write('hello,world\n1,2\n')

        bp.remove_bom(filepath)

        # Read back and check BOM is removed
        with open(filepath, 'rb') as f:
            content = f.read()
        assert not content.startswith(b'\xef\xbb\xbf')


def test_array_to_fits_and_get_fits_data():
    # Save current directory to restore later
    original_dir = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
            bp.array_to_fits(arr)

            assert os.path.exists('output.fits')

            # Read back using get_fits_data
            result = bp.get_fits_data('output.fits', 1)
            assert np.array_equal(result, arr)
        finally:
            os.chdir(original_dir)


def test_read_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.json')
        data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}

        with open(filepath, 'w') as f:
            json.dump(data, f)

        result = bp.read_json(filepath)
        assert result == data


def test_batch_list():
    result = list(bp.batch_list([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

    result = list(bp.batch_list([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

    result = list(bp.batch_list([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

    result = list(bp.batch_list([], 2))
    assert result == []


def test_combine_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        img1_path = os.path.join(tmpdir, 'img1.png')
        img2_path = os.path.join(tmpdir, 'img2.png')
        output_path = os.path.join(tmpdir, 'combined.png')

        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (50, 50), color='blue')
        img1.save(img1_path)
        img2.save(img2_path)

        bp.combine_images(2, 10, [img1_path, img2_path], output_path)

        assert os.path.exists(output_path)
        result = Image.open(output_path)
        # With 2 columns, 50px images, and 10px spacing: width = 50*2 + 10 - 10 = 100
        assert result.width == 110


def test_print_execution_time(capsys):
    bp.print_execution_time(0, 65)
    captured = capsys.readouterr()
    assert 'Execution time: 0:01:05' in captured.out


def test_pickle_and_unpickle():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.pkl')
        data = {'key': 'value', 'list': [1, 2, 3]}

        bp.pickle(filepath, data)
        assert os.path.exists(filepath)

        result = bp.unpickle(filepath)
        assert result == data


def test_unpickle_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test pickle files
        bp.pickle(os.path.join(tmpdir, 'data_1.pkl'), {'id': 1})
        bp.pickle(os.path.join(tmpdir, 'data_2.pkl'), {'id': 2})
        bp.pickle(os.path.join(tmpdir, 'other_3.pkl'), {'id': 3})

        # Test unpickle all with prefix
        result = bp.unpickle_all(tmpdir, prefix='data_')
        assert len(result) == 2

        # Test with limit
        result = bp.unpickle_all(tmpdir, prefix='data_', limit=1)
        assert len(result) == 1

        # Test without prefix (all files)
        result = bp.unpickle_all(tmpdir)
        assert len(result) == 3


def test_create_directory_if_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, 'new_subdir', 'nested')
        assert not os.path.exists(new_dir)

        bp.create_directory_if_not_exists(new_dir)
        assert os.path.exists(new_dir)

        # Should not raise if already exists
        bp.create_directory_if_not_exists(new_dir)
        assert os.path.exists(new_dir)


def test_clear_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files and subdirectories
        with open(os.path.join(tmpdir, 'file1.txt'), 'w') as f:
            f.write('test')
        with open(os.path.join(tmpdir, 'file2.txt'), 'w') as f:
            f.write('test')

        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        with open(os.path.join(subdir, 'nested.txt'), 'w') as f:
            f.write('test')

        bp.clear_directory(tmpdir)

        assert os.path.exists(tmpdir)
        assert len(os.listdir(tmpdir)) == 0


def test_scientific_notation_string():
    result = bp.scientific_notation_string(1500)
    assert '1.5' in result
    assert '10^{3}' in result

    result = bp.scientific_notation_string(0.0025)
    assert '2.5' in result
    assert '10^{-3}' in result


def test_delete_if_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.txt')

        # Should not raise if file doesn't exist
        bp.delete_if_exists(filepath)

        # Create file and delete it
        with open(filepath, 'w') as f:
            f.write('test')
        assert os.path.exists(filepath)

        bp.delete_if_exists(filepath)
        assert not os.path.exists(filepath)


def test_get_today_str():
    result = bp.get_today_str()
    expected = str(datetime.date.today())
    assert result == expected
    # Check format YYYY-MM-DD
    assert len(result) == 10
    assert result[4] == '-'
    assert result[7] == '-'
