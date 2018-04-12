import numpy as np
from scipy.spatial.distance import cdist


class MinimumDistanceClassifier(object):
    """
    Codification of generic minimum distance classifier in a scikit-learn friendly format.
    """
    def __init__(self, metric='correlation'):
        """
        Arguments: 
            metric -- what distance metric to use (consistent with scipy.spatial.distance.cdist)
        """
        self.metric = metric

    def initialize(self, n_classes, n_features, labeset, val=None, n=None):
        self.n_classes = n_classes
        self.n_features = n_features
        self.labelset = labelset
        self._mu = np.zeros((self.n_classes, self.n_features))
        self._n_samples = np.zeros((self.n_classes,)).astype(np.int)
        if val is not None:
            assert val.shape == self.mu.shape
            assert n.dtype == self._n_samples.dtype
            assert n.shape == self._n_samples.shape
            assert (n >= 0).all()
            self._mu += val
            self._n_samples += n

    def fit(self, X, y):
        labelset = np.unique(y)
        n_classes = len(labelset)
        n_features = X.shape[1]
        self.initialize(n_classes, n_features, labelset, val=None, n=None)
        self.partial_fit(X, y)

    def partial_fit(self, X, y, safe=True):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape == (len(y), self.n_features), (X.shape, (len(y), self.n_features))
        assert set(y) <= set(self.labelset), (set(y), set(self.labelset))
        if safe:
            uy = np.unique(y)
            assert (uy == self.labelset).all()
        for vi, v in enumerate(self.labelset):
            Xv = X[y == v]
            nv = float(Xv.shape[0])
            if nv > 0:
                Xvm = Xv.mean(0)
                ns = self._n_samples[vi]
                self._mu[vi] = (ns / (ns + nv)) * self._mu[vi] + (nv / (ns + nv)) * Xvm
                self._n_samples[vi] += int(nv)

    @property
    def coef_(self):
        c = self._mu
        return c

    @property
    def weights(self):
        c = self.coef_
        def weightfunc(x):
            return (x - x.mean(1)[:, np.newaxis]) / (x.std(1)[:, np.newaxis])
        return weightfunc(c)

    def decision_function(self, X):
        return cdist(X, self.coef_, metric=self.metric)

    def predict(self, X):
        assert X.ndim == 2
        assert X.shape[1] == self.n_features
        decs = self.decision_function(X)
        return self.labelset[decs.argmin(1)]