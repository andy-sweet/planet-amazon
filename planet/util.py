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
import os, csv, zipfile

# Third party
import numpy
import skimage.io, skimage.transform
import sklearn.model_selection
import plotly
import wget

_this_dir = os.path.dirname(__file__)
_repo_dir = os.path.join(_this_dir, "..")
data_dir = os.path.join(_repo_dir, "data")

data_url = 'https://storage.googleapis.com/planet-amazon'
train_tags_name = 'train_v2.csv'
train_images_name = 'train-jpg'

train_tags_file_path = os.path.join(data_dir, train_tags_name)
train_images_dir_path = os.path.join(data_dir, train_images_name)


def get_train_data(num_samples=None, image_size=None):
    all_tags = get_train_tags()

    if num_samples is None:
        num_samples = len(all_tags)

    train_inds, test_inds = next(split_data(num_samples, 2))

    tag_indices = get_tag_indices(all_tags)
    labels = tags_to_labels(all_tags, tag_indices)[train_inds, :]

    all_names = list(all_tags.keys())
    train_names = [all_names[ind] for ind in train_inds]
    images = read_images(train_images_dir_path, train_names, out_size=image_size)

    return (images, labels)


def download_train_tags(force=False):
    """ Download the training tags from the public remote location and extract them.

    Keyword Arguments
    =================
    force : bool
        If true, overwrite existing data if it already exists.
    """
    if not os.path.exists(train_tags_file_path) or force:
        train_tags_url = '{}/{}.zip'.format(data_url, train_tags_name)
        os.makedirs(data_dir, exist_ok=True)
        zip_file_path = wget.download(train_tags_url, out=data_dir)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(data_dir)


def download_train_images(force=False):
    """ Download the training images from the public remote location and extract them.

    Keyword Arguments
    =================
    force : bool
        If true, overwrite existing data if it already exists.
    """
    if not os.path.exists(train_images_dir_path) or force:
        train_images_url = '{}/{}.zip'.format(data_url, train_images_name)
        os.makedirs(data_dir, exist_ok=True)
        zip_file_path = wget.download(train_images_url, out=data_dir)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(data_dir)


def get_train_tags(force=False):
    """ Download (if needed) and read the training tags.

    Keyword Arguments
    =================
    force : bool
        If true, overwrite existing data if it already exists.
    """
    download_train_tags(force)
    return read_tags(train_tags_file_path)



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
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            tags[row['image_name']] = row['tags'].split()
    return tags


def read_images(image_dir, names, out_size=None):
    """ Reads the images with the given names.
    """
    num_images = len(names)
    image = skimage.io.imread(os.path.join(image_dir, names[0] + '.jpg'))
    dtype = image.dtype;
    if out_size is None:
        images = numpy.empty((num_images, 256, 256, 3), dtype=dtype)
    else:
        images = numpy.empty((num_images, out_size[0], out_size[1], 3), dtype=dtype)

    for index, name in enumerate(names):
        image = skimage.io.imread(os.path.join(image_dir, name + '.jpg'))
        if out_size is None:
            images[index, :, :, :] = image
        else:
            images[index, :, :, :] = skimage.transform.resize(image, out_size, mode='reflect').astype(dtype)

    return images


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


def get_tag_indices(tags):
    """ Associate an index with a set of tags.

    Arguments
    ---------
    tags : dict
        Maps sample name to tags.

    Returns
    -------
    tag_indices : dict
        Maps tag name to an index in the output matrix.
    """
    tag_counts = count_tags(tags);
    tag_names = tag_counts.keys()
    return {name : index for (index, name) in enumerate(tag_names)}


def tags_to_labels(tags, tag_indices):
    """ Converts dictionary of tags to matrix of binary labels.

    Arguments
    ---------
    tags : dict
        Maps sample name to tags.
    tag_indices : dict
        Maps tag name to an index in the output matrix.

    Returns
    -------
    labels : :class:`numpy.ndarray`, (N, K), bool
        The matrix of labels.
    """
    num_samples = len(tags)
    num_labels = len(tag_indices)

    labels = numpy.zeros((num_samples, num_labels), dtype=bool)
    for sample_index, sample_name in enumerate(tags.keys()):
        for tag in tags[sample_name]:
            labels[sample_index, tag_indices[tag]] = 1
    return labels


def split_data(num_samples, num_splits):
    """ Yields a split of data into train and test indices.
    """

    kf = sklearn.model_selection.KFold(n_splits=num_splits, random_state=0);
    return kf.split(range(num_samples))


def make_bar_plot(x, y, title):
    """ Makes a bar chart with a title.
    """
    return plotly.graph_objs.Figure(
            data=[plotly.graph_objs.Bar(x=list(x), y=list(y))],
            layout=plotly.graph_objs.Layout(title=title)
    )


def make_bar_group_plot(x, Y, groups, colors, title):
    """ Makes a grouped bar chart with a title.
    """
    data = []
    for i in range(len(groups)):
        data.append(plotly.graph_objs.Bar(
                x=list(x),
                y=list(Y[i, :]),
                name=groups[i],
                marker={'color' : colors[i]}
        ))

    return plotly.graph_objs.Figure(
            data=data,
            layout=plotly.graph_objs.Layout(title=title, barmode='group')
    )
