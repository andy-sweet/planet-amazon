# Built-in
import os, sys

# Related
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import planet.util

import numpy


def test_split_data():
    split_index = 0
    for train, test in planet.util.split_data(10, 2):
        if split_index == 0:
            numpy.testing.assert_equal(train, range(5, 10))
            numpy.testing.assert_equal(test, range(0, 5))
        elif split_index == 1:
            numpy.testing.assert_equal(train, range(0, 5))
            numpy.testing.assert_equal(test, range(5, 10))
        split_index += 1
    assert split_index == 2


def test_get_train_data():
    [images, labels] = planet.util.get_train_data(4, image_size=(8, 8))
    assert images.shape == (2, 8, 8, 3)
    assert images.dtype == numpy.uint8
    assert labels.shape == (2, 17)


def test_download_train_tags():
    planet.util.download_train_tags()
    assert os.path.exists(planet.util.train_tags_file_path)


def test_download_train_images():
    planet.util.download_train_images()
    assert os.path.exists(planet.util.train_images_dir_path)
    assert os.path.exists(os.path.join(planet.util.train_images_dir_path, 'train_0.jpg'))
