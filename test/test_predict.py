import os, sys, tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import planet.predict

import numpy.testing


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


def test_vgg19_cnn():
    """ Check that the VGG19 CNN is trained and can predict the right shape/class.
    """
    num_train = 4
    num_test = 3
    num_rows = 224
    num_cols = 224
    num_channels = 3
    num_labels = 2
    images = numpy.random.randint(256, size=(num_train, num_rows, num_cols, num_channels), dtype='uint8')
    labels = numpy.random.randint(2, size=(num_train, num_labels), dtype='bool')
    cnn = planet.predict.Vgg19Cnn.from_data(images, labels, 4)
    test_images = numpy.random.randint(256, size=(num_test, num_rows, num_cols, num_channels), dtype='uint8')
    pred_labels = cnn.predict(test_images)
    assert pred_labels.shape == (num_test, num_labels)

    out_file = tempfile.NamedTemporaryFile()
    cnn.write(out_file.name)

    read_cnn = planet.predict.Vgg19Cnn.from_file(out_file.name)
    numpy.testing.assert_equal(cnn.top.get_weights(), read_cnn.top.get_weights())


def test_vgg19_cnn_train():
    """ Check that the VGG19 CNN training creates a file with identical weights.
    """
    out_file = tempfile.NamedTemporaryFile()
    cnn = planet.predict.Vgg19Cnn.train(out_file.name, num_samples=2)
    read_cnn = planet.predict.Vgg19Cnn.from_file(out_file.name)
    numpy.testing.assert_equal(cnn.top.get_weights(), read_cnn.top.get_weights())


def test_resnet50_cnn():
    """ Check that the ResNet50 CNN is trained and can predict the right shape/class.
    """
    num_train = 4
    num_test = 3
    num_rows = 224
    num_cols = 224
    num_channels = 3
    num_labels = 2
    images = numpy.random.randint(256, size=(num_train, num_rows, num_cols, num_channels), dtype='uint8')
    labels = numpy.random.randint(2, size=(num_train, num_labels), dtype='bool')
    cnn = planet.predict.ResNet50Cnn.from_data(images, labels, 4)
    test_images = numpy.random.randint(256, size=(num_test, num_rows, num_cols, num_channels), dtype='uint8')
    pred_labels = cnn.predict(test_images)
    assert pred_labels.shape == (num_test, num_labels)

    out_file = tempfile.NamedTemporaryFile()
    cnn.write(out_file.name)

    read_cnn = planet.predict.ResNet50Cnn.from_file(out_file.name)
    numpy.testing.assert_equal(cnn.top.get_weights(), read_cnn.top.get_weights())


def test_resnet50_cnn_train():
    """ Check that the ResNet50 CNN training creates a file with identical weights.
    """
    out_file = tempfile.NamedTemporaryFile()
    cnn = planet.predict.ResNet50Cnn.train(out_file.name, num_samples=2)
    read_cnn = planet.predict.ResNet50Cnn.from_file(out_file.name)
    numpy.testing.assert_equal(cnn.top.get_weights(), read_cnn.top.get_weights())
