""" Some utility functions for loading and exploring the data.

Nomenclature
------------
tag : The name of a label to predict. E.g. "haze".
label : The non-negative integer value associated with a tag. E.g. 3.
sample : The name associated with a sample. E.g. "train_1"
N : The number of samples.
Y : The number of rows in an image.
X : The number of columns in an image.
C : The number of channels in an image.
K : The number of labels.
"""

# Built-in
import os, csv

# Third party
import numpy as np
import skimage.io

# Some module constants related to data paths
default_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")


def read_tags(csv_path):
    """ Read tags from a CSV file into a map.

    Arguments
    ---------
    csv_path : str
        The path of the tag CSV file.

    Returns
    -------
    dict
        Maps sample name to a vector of tags.
    """
    tags = {}
    with open(csv_path,"r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            tags[row["image_name"]] = row["tags"].split()
    return tags


def count_tags(tags):
    """ Count of the occurrences of tags.

    Arguments
    ---------
    tags : dict
        Maps sample name to tags.

    Returns
    -------
    dict
        Maps tag to count.
    """
    counts = {}
    for tag_list in tags.values():
        for tag in tag_list:
            if tag in counts:
                counts[tag] += 1
            else:
                counts[tag] = 1
    return counts
