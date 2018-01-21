""" Tests the utility functions.
"""

# Built-in
import os, sys

# Related
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import planet.util


def test_download_train_tags():
    """ tests downloading the training tags.
    """
    planet.util.download_train_tags()
    assert os.path.exists(planet.util.train_tags_file_path)


def test_download_train_images():
    """ Tests downloading the training images.
    """
    planet.util.download_train_images()
    assert os.path.exists(planet.util.train_images_dir_path)
    assert os.path.exists(os.path.join(planet.util.train_images_dir_path, 'train_0.jpg'))
