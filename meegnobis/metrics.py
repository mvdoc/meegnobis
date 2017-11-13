"""Module containing metrics"""
import numpy as np
from .utils import _npairs, _get_unique_targets
from scipy.spatial.distance import cdist


class BaseMetric(object):
    """A basic metric class"""
    def __init__(self):
        # whether the metric is symmetric or not
        self._symmetric = False
        # whether the metric is vectorized (i.e., doesn't need to be run
        # separately for each target)
        self._vectorized = False
        # other values
        self.data_train = None
        self.targets_train = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    @property
    def is_symmetric(self):
        return self._symmetric

    @property
    def is_vectorized(self):
        return self._vectorized


class CDistMetric(BaseMetric):
    def __init__(self, metric):
        super(CDistMetric, self).__init__()
        self._symmetric = True
        self._vectorized = True
        self._unique_targets_train = None
        self._n_pairwise_targets_train = None
        self.metric = lambda x, y: cdist(x, y, metric=metric)

    def fit(self, X, y):
        self.data_train = X
        self.targets_train = y
        self._unique_targets_train = _get_unique_targets(self.targets_train,
                                                         self.targets_train)
        self._n_pairwise_targets_train = _npairs(
            len(self._unique_targets_train))

    def predict(self, X):
        if len(self._unique_targets_train) != X.shape[0]:
            raise ValueError(
                "Training set had {0} unique targets, and I am "
                "expecting X to have the same number of "
                "samples, but got {1}".format(len(self._unique_targets_train),
                                              X.shape[0]))
        dist = self.metric(self.data_train, X)
        # make it symmetric
        dist += dist.T
        dist /= 2
        return dist[np.triu_indices_from(dist)]

    def score(self, X, y):
        return self.predict(X)


CDIST_METRICS_ = [
    'cityblock',
    'correlation',
    'cosine',
    'dice',
    'euclidean',
    'hamming',
    'sqeuclidean'
]

CDIST_METRICS = {
    metric: CDistMetric(metric=metric) for metric in CDIST_METRICS_
}
