""" Prediction methods.
"""

# Local
import planet.util

# Built-in

# Third party
import numpy
import sklearn.neighbors, sklearn.metrics


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


class NearestNeighbors:

    def __init__(self, X, Y, num_neighbors):
        """ Creates a new nearest neighbors model.
        """
        self.impl = sklearn.neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
        self.impl.fit(Y, X)
        self.K = Y.shape[1]


    def predict(self, Y):
        """ Predicts labels from observed data.
        """
        return self.impl.predict(Y) > 0.5


    def predict_prob(self, Y):
        """ Predicts the probability of each label based on the training data.
        """
        return self.impl.predict_proba(Y)


def f2_score(pred_labels, true_labels, avg_type):
    """ Compute the average F2 score.
    """
    return sklearn.metrics.fbeta_score(true_labels, pred_labels, beta=2, average=avg_type)


def plot_scores(pred_labels, true_labels, label_names, classifier, file_path):
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

    planet.util.plot_bar_group(label_names, scores, groups, colors, title, file_path)

    return [recall_avg, precision_avg, f2_avg]
