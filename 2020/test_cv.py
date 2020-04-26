import numpy as np
from balanced_kfold import BalancedKFold


# smoke test
def test_get_balanced_folds():
    Y = np.random.randn(100, 1)
    X = np.random.randn(100, 2)
    bf = BalancedKFold(4)
    _ = bf.split(X, Y)
