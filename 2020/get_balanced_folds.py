
"""
This code implements a cross-validation splitter with an sklearn
API, which returns cross-validation folds that are balanced for
the distribution of the outcome variable.  This is meant for use
in regression, where unbalanced folds can lead to bad behavior.
"""

from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
import numpy as np


class BalancedKFold:
    """
    This class implments a Balanced Kfold splitter with an sklearn
    interface.  The resulting folds will be balanced in the
    mean of the Y value. see Kohavi, 1995

    Parameters
    ---------------
    nfolds: int, number of folds to split (default: 5)
    pthresh: float (0 < pthresh < 1), threshold for ANOVA comparing
    means across groups. Higher threshold gives better balancing
    (default: 0.8)
    """
    def __init__(self, nfolds=5, pthresh=0.8, verbose=False):
        self.nfolds = nfolds
        self.pthresh = pthresh
        self.verbose = verbose

    def split(self, X, Y, seed=None, max_splits=1000):
        """
        splitting function

        Parameters
        -----------
        X: design matrix (not actually needed but taken for consistency)
        Y: outcome variable
        seed: random seed (default: None, uses system time)
        max_splits: int, maximum number of splits to try before failing
        """

        np.random.seed(seed)
        nsubs = len(Y)
        # cycle through until we find a split that is good enough
        runctr = 0
        best_pval = 0
        while True:
            runctr += 1
            cv = KFold(n_splits=self.nfolds, shuffle=True)

            idx = np.zeros((nsubs, self.nfolds))  # this is the design matrix
            folds = []
            ctr = 0
            # create design matrix for anova across folds
            for train, test in cv.split(Y):
                idx[test, ctr] = 1
                folds.append([train, test])
                ctr += 1

            # fit anova model, comparing means of Y across folds
            lm_y = OLS(Y - np.mean(Y), idx).fit()

            if lm_y.f_pvalue > best_pval:
                best_pval = lm_y.f_pvalue
                best_folds = folds

            if lm_y.f_pvalue > self.pthresh:
                if self.verbose:
                    print(lm_y.summary())
                return iter(folds)

            if runctr > max_splits:
                print('no sufficient split found, returning best (p=%f)' % best_pval) # noqa
                return iter(best_folds)


if __name__ == "__main__":
    Y = np.random.randn(100, 1)
    bf = BalancedKFold(4, verbose=True)
    s = bf.split(Y, Y)
