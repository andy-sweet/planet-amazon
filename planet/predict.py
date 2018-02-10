""" Prediction methods.
"""

# Local
import planet.util

# Built-in

# Third party
import numpy
import sklearn.neighbors, sklearn.metrics, sklearn.model_selection
import tqdm
import keras.models, keras.layers


def random(N, K, seed=0):
    """ Predict a label matrix completely at random.

    Arguments
    ---------
    N : int
        The number of samples to predict.
    K : int
        The number of labels.

    Keyword Arguments
    -----------------
    seed : int
        The seed used for random number generation.

    Returns
    -------
    :class:`numpy.ndarray`, (N, K), bool
        The predicted labels.
    """

    numpy.random.seed(seed)
    return numpy.random.random((N, K)) < 0.5


def empirical(N, label_probs):
    """ Predicts a label matrix using the empirical probabilities of labels.

    Arguments
    ---------
    N : int
        The number of samples to predict.
    label_probs : :class:`numpy.ndarray`, (1, K), float
        The probability of each label.

    Returns
    -------
    :class:`numpy.ndarray`, (N, K), bool
        The predicted labels.
    """

    return numpy.tile(label_probs, (N, 1)) < 0.5


def empirical_random(N, label_probs, seed=0):
    """ Predicts a label matrix using the empirical probabilities of labels.

    Arguments
    ---------
    N : int
        The number of samples to predict.
    label_probs : :class:`numpy.ndarray`, (1, K), float
        The probability of each label.

    Keyword Arguments
    -----------------
    seed : int
        The seed used for random number generation.

    Returns
    -------
    :class:`numpy.ndarray`, (N, K), bool
        The predicted labels.
    """

    numpy.random.seed(seed)
    return numpy.random.rand(N, label_probs.size) < label_probs


def split_data(num_samples, num_splits):
    """ Yields a split of data into train and test indices.
    """

    kf = sklearn.model_selection.KFold(n_splits=num_splits, random_state=0);
    return kf.split(range(num_samples))


def flatten_images(images):
    return numpy.reshape(images, (images.shape[0], -1))


def score_k_nearest_neighbors(images, labels, ks, num_splits):
    """ Scores k-nearest neighbors with different values of k.
    """

    num_samples = images.shape[0]
    num_ks = len(ks)
    scores = numpy.zeros((num_splits, num_ks))
    with tqdm.tqdm(total=(num_splits * num_ks)) as progress:
        for split_index, (train, test) in enumerate(split_data(num_samples, num_splits)):
            for k_index, k in enumerate(ks):
                knn = KNearestNeighbors(images[train, :, :, :], labels[train, :], k)
                pred_labels = knn.predict(images[test, :, :, :])
                scores[split_index, k_index] = f2_score(pred_labels, labels[test, :], 'micro')
                progress.update((split_index * num_ks) + k_index + 1)

    return numpy.mean(scores, axis=0)


class KNearestNeighbors:

    def __init__(self, images, labels, k):
        """ Creates a new nearest neighbors model.
        """
        self.impl = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        self.impl.fit(flatten_images(images), labels)


    def predict(self, images):
        """ Predicts labels from observed data.
        """
        return self.impl.predict(flatten_images(images))


class ConvNeuralNetwork:

    def __init__(self, images, labels, num_hidden, num_filters, filter_sizes, batch_size=128, num_epochs=1):
        """ Creates a convolutional neural network.

        Arguments
        ---------
        images : array-like (N, R, C, D)
            The input images with which to train.
        labels : array-like (N, K)
            The output labels with which to train.
        num_hidden : int > 0
            Number of hidden nodes in penultimate layer.
        num_filters : array-like (F,)
            Number of filters in each layer (length determines number of layers).
        filter_sizes : array-like (F,)
            Sizes of filters in each layer (length should match that of num_filters).
        """

        (N, K) = labels.shape
        (N, R, C, D) = images.shape
        F = len(num_filters)

        self.impl = keras.models.Sequential()
        for f in range(F):
            self.impl.add(keras.layers.convolutional.Conv2D(
                num_filters[f], kernel_size=filter_sizes[f],
                activation='relu', input_shape=(R, C, D), padding='same')
            )
            self.impl.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            self.impl.add(keras.layers.Dropout(0.25))

        self.impl.add(keras.layers.Flatten())
        self.impl.add(keras.layers.Dense(num_hidden, activation='relu'))
        self.impl.add(keras.layers.Dropout(0.5))
        self.impl.add(keras.layers.Dense(K, activation='sigmoid'))
        optimizer = keras.optimizers.SGD()
        self.impl.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.impl.summary()
        self.impl.fit(images, labels, batch_size=batch_size, epochs=num_epochs, verbose=2)


    @classmethod
    def read(cls, file_path):
        self.impl = keras.models.load_model(file_path)


    def write(self, file_path):
        self.impl.save(file_path)


    def predict(self, images):
        """ Predicts labels from observed data.
        """

        return self.impl.predict(images) > 0.5


def f2_score(pred_labels, true_labels, avg_type):
    """ Compute the average F2 score.
    """
    return sklearn.metrics.fbeta_score(true_labels, pred_labels, beta=2, average=avg_type)


def make_scores_plot(pred_labels, true_labels, label_names, classifier):
    """ Computes and plots the scores associated with a classifier.
    """

    recall = sklearn.metrics.recall_score(true_labels, pred_labels, average=None)
    precision = sklearn.metrics.precision_score(true_labels, pred_labels, average=None)
    f2 = sklearn.metrics.fbeta_score(true_labels, pred_labels, beta=2, average=None)
    scores = numpy.array([recall, precision, f2])

    avg_type = 'micro'
    recall_avg = sklearn.metrics.recall_score(true_labels, pred_labels, average=avg_type)
    precision_avg = sklearn.metrics.precision_score(true_labels, pred_labels, average=avg_type)
    f2_avg = f2_score(pred_labels, true_labels, avg_type)

    title = 'Scores for {} Classifier. Recall={:.2f}, Precision={:.2f}, F2={:.2f}'.format(classifier, recall_avg, precision_avg, f2_avg)
    colors = ['rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)']
    groups = ['recall', 'precision', 'f2']

    return planet.util.make_bar_group_plot(label_names, scores, groups, colors, title)
