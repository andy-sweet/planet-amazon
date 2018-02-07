import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import planet.predict

import numpy.testing

def test_split_data():
    """ Check a two way split of data.
    """
    split_index = 0
    for train, test in planet.predict.split_data(10, 2):
        if split_index == 0:
            numpy.testing.assert_equal(train, range(5, 10))
            numpy.testing.assert_equal(test, range(0, 5))
        elif split_index == 1:
            numpy.testing.assert_equal(train, range(0, 5))
            numpy.testing.assert_equal(test, range(5, 10))
        split_index += 1
    assert split_index == 2


def test_1_nearest_neighbor():
    """ Check we get the same neighbor back for perfect prediction.
    """
    images = numpy.array([[[[0]]], [[[1]]], [[[2]]], [[[3]]]])
    labels = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    knn = planet.predict.KNearestNeighbors(images, labels, 1)
    pred_labels = knn.predict(images)
    numpy.testing.assert_equal(pred_labels, labels)


def test_3_nearest_neighbors():
    """ Check non-perfect prediction with 3 neighbors.
    """
    images = numpy.array([[[[0]]], [[[1]]], [[[2]]], [[[3]]]])
    labels = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    knn = planet.predict.KNearestNeighbors(images, labels, 3)
    pred_labels = knn.predict(images)
    numpy.testing.assert_equal(pred_labels, [[0, 0], [0, 0], [1, 1], [1, 1]])


def test_score_k_nearest_neighbors():
    """ Just check we get the right number of scores.
    """
    images = numpy.array([[[[0]]], [[[1]]], [[[2]]], [[[3]]]])
    labels = numpy.array([[1, 1], [1, 1], [1, 1], [1, 1]])
    scores = planet.predict.score_k_nearest_neighbors(images, labels, [1, 3], 4)
    assert scores.shape == (2,)
