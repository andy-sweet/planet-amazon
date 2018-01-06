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
import csv

# Third party
import numpy

import plotly
plotly.offline.init_notebook_mode(connected=True)

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


def plot_bar(x, y, title, file_path):
    """ Plots a bar chart with a title and writes to a file.
    """
    fig = plotly.graph_objs.Figure(
            data=[plotly.graph_objs.Bar(x=list(x), y=list(y))],
            layout=plotly.graph_objs.Layout(title=title)
    )
    plotly.offline.iplot(fig, filename=file_path)


def plot_bar_group(x, Y, groups, colors, title, file_path):
    """ Plots a grouped bar chart.
    """
    data = []
    for i in range(len(groups)):
        data.append(plotly.graph_objs.Bar(
                x=list(x),
                y=list(Y[i, :]),
                name=groups[i],
                marker={'color' : colors[i]}
        ))

    fig = plotly.graph_objs.Figure(
            data=data,
            layout=plotly.graph_objs.Layout(title=title, barmode='group')
    )

    plotly.offline.iplot(fig, filename=file_path)
