""" Prediction methods.
"""

# Local
import planet.util

# Built-in
import warnings

# Third party
import numpy
import sklearn.neighbors, sklearn.metrics
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


def flatten_images(images):
    return numpy.reshape(images, (images.shape[0], -1))


def score_k_nearest_neighbors(images, labels, ks, num_splits):
    """ Scores k-nearest neighbors with different values of k.
    """
    num_samples = images.shape[0]
    num_ks = len(ks)
    scores = numpy.zeros((num_splits, num_ks))
    with tqdm.tqdm(total=(num_splits * num_ks)) as progress:
        for split_index, (train, test) in enumerate(planet.util.split_data(num_samples, num_splits)):
            for k_index, k in enumerate(ks):
                knn = KNearestNeighbors(images[train, :, :, :], labels[train, :], k)
                pred_labels = knn.predict(images[test, :, :, :])
                scores[split_index, k_index] = f2_score(pred_labels, labels[test, :], 'micro')
                progress.update(1)

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


class Cnn(object):

    def __init__(self):
        self.bottom = None
        self.top = None

    @classmethod
    def from_data(cls, images, labels, batch_size=64, num_epochs=3):
        self = cls()
        (N, K) = labels.shape
        assert images.shape[0] == N

        self._init_bottom()
        responses = self.bottom.predict(images)

        self.top = keras.models.Sequential()
        self.top.add(keras.layers.Flatten(input_shape=responses.shape[1:]))
        self.top.add(keras.layers.Dense(K, activation='sigmoid'))
        self.top.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.top.fit(responses, labels, batch_size=batch_size, epochs=num_epochs, verbose=2)
        return self

    @classmethod
    def from_file(cls, file_path):
        self = cls()
        self._init_bottom()
        self.top = keras.models.load_model(file_path)
        return self

    @classmethod
    def train(cls, file_path, batch_size=64, num_epochs=3, num_samples=None):
        [images, labels] = planet.util.get_train_data(num_samples=num_samples, image_size=(224, 224))
        self = cls.from_data(images, labels, batch_size=batch_size, num_epochs=num_epochs)
        self.write(file_path)
        return self

    def _init_bottom(self):
        raise NotImplementedError()

    def write(self, file_path):
        self.top.save(file_path)

    def predict(self, images):
        responses = self.bottom.predict(images, verbose=1)
        return self.top.predict(responses, verbose=1) > 0.5


class Vgg19Cnn(Cnn):

    def _init_bottom(self):
        self.bottom = keras.applications.vgg19.VGG19(include_top=False)


class ResNet50Cnn(Cnn):

    def _init_bottom(self):
        self.bottom = keras.applications.resnet50.ResNet50(include_top=False)


def precision_score(pred_labels, true_labels, avg_type):
    """ Compute the precision.
    """
    warnings.simplefilter('ignore')
    return sklearn.metrics.precision_score(true_labels, pred_labels, average=avg_type)


def recall_score(pred_labels, true_labels, avg_type):
    """ Compute the recall.
    """
    warnings.simplefilter('ignore')
    return sklearn.metrics.recall_score(true_labels, pred_labels, average=avg_type)


def f2_score(pred_labels, true_labels, avg_type):
    """ Compute the average F2 score.
    """
    warnings.simplefilter('ignore')
    return sklearn.metrics.fbeta_score(true_labels, pred_labels, beta=2, average=avg_type)


def make_scores_plot(pred_labels, true_labels, label_names, classifier):
    """ Computes and plots the scores associated with a classifier.
    """

    recall = recall_score(pred_labels, true_labels, None)
    precision = precision_score(pred_labels, true_labels, None)
    f2 = f2_score(pred_labels, true_labels, None)

    scores = numpy.array([recall, precision, f2])

    avg_type = 'micro'
    recall_avg = recall_score(pred_labels, true_labels, avg_type)
    precision_avg = precision_score(pred_labels, true_labels, avg_type)
    f2_avg = f2_score(pred_labels, true_labels, avg_type)

    title = 'Scores for {} Classifier. Recall={:.2f}, Precision={:.2f}, F2={:.2f}'.format(
            classifier, recall_avg, precision_avg, f2_avg)
    colors = ['rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)']
    groups = ['recall', 'precision', 'f2']

    return planet.util.make_bar_group_plot(label_names, scores, groups, colors, title)
