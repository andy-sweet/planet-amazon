""" Tests the utility functions.
"""

# Built-in
import os, sys

# Related
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import planet.util


def test_download_train_tags():
    """ Tests downloading the training tags.
    """
    planet.util.download_train_tags()
    assert os.path.exists(planet.util.train_tags_file_path)
