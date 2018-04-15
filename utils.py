import numpy as np


def featurewise_norm(data, fmean=None, fvar=None):
    """perform a whitening-like normalization operation on the data, feature-wise
       Assumes data = (K, M) matrix where K = number of stimuli and M = number of features
    """
    if fmean is None:
        fmean = data.mean(0)
    if fvar is None:
        fvar = data.std(0)
    data = data - fmean  #subtract the feature-wise mean of the data
    data = data / np.maximum(fvar, 1e-5)  #divide by the feature-wise std of the data
    return data, fmean, fvar


def get_off_diagonal(mat):
    n = mat.shape[0]
    i0, i1 = np.triu_indices(n, 1)
    i2, i3 = np.tril_indices(n, -1)
    return np.concatenate([mat[i0, i1], mat[i2, i3]])


def spearman_brown(uncorrected, multiple):
    numerator = multiple * uncorrected
    denominator = 1 + (multiple - 1) * uncorrected
    return numerator / denominator
