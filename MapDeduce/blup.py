"""BLUP: Best linear unbiased predictions of a linear mixed model."""

import numpy as np
import pandas as pd

from limix_legacy.modules.varianceDecomposition import VarianceDecomposition

from sklearn.model_selection import KFold

from scipy.spatial.distance import euclidean

from tqdm import tqdm


class LmmBlup(object):
    """Best linear unbiased predictions of a linear mixed model."""

    def __init__(self, Y, **kwargs):
        """
        At least one of F and K must be specified.

        @param Y: Response. n x p ndarray

            Optional kwargs:

        @param F: Fixed effects predictors. n x s ndarray
        @param K: Random effects predictors. n x r ndarray
        @param A: Trait design matrix. p x p ndarray
        """
        self.Y = Y
        self.F = kwargs.pop("F", None)
        self.K = kwargs.pop("K", None)
        self.A = kwargs.pop("A", None)

        if self.F is None and self.K is None:
            raise ValueError("At least one of F and K must be specified.")

        if self.A is None and self.F is not None:
            raise ValueError("Must specify design matrix")

    def predict(self, train, test):
        """
        Train LMM using values in the training set, and return predicitons
        for responses in the test set.

        @param train. n x 1 ndarray. Indexes of rows to use as the train set
        @param test. n x 1 ndarray. Indexes of rows to use as the test set
        """
        vc = VarianceDecomposition(Y=self.Y[train])

        vc.setTestSampleSize(test.shape[0])

        if self.F is not None:
            vc.addFixedEffect(
                F=self.F[train],
                Ftest=self.F[test],
                A=self.A)

        if self.K is not None:
            vc.addRandomEffect(
                K=self.K[train, :][:, train],
                Kcross=self.K[train, :][:, test])

        vc.addRandomEffect(is_noise=True)

        vc.optimize()

        return vc.predictPhenos()

    def predict_kfolds(self, n_splits, random_state=1234, progress_bar=True):
        """
        Predict k test folds of the data having trained on training folds.

        @param n_splits: Int. Number of folds.
        @param random_state: Int. Used to initialize random state.
        @param progress_bar: Bool. Show progress bar for each fold.

        Attaches kfold_predictions and kfold_error attributes to self. These
        are dictionaries. Keys are each fold. Predictions contain the
        predicted response variables. Errors contain the distance between
        the predictions and the test set.
        """
        kf = KFold(
            n_splits=n_splits,
            random_state=random_state,
            shuffle=True)

        self.kfold_predictions = {}
        self.kfold_error = {}

        iterable = enumerate(kf.split(self.Y))

        if progress_bar:
            iterable = tqdm(iterable)

        for i, (train, test) in iterable:

            p = self.predict(
                train=train,
                test=test)

            train_set = self.Y[train]
            test_set = self.Y[test]

            self.kfold_predictions[i] = dict(
                train=train_set,
                test=test_set,
                prediction=p)

            distance = np.empty(test_set.shape[0])

            for j in range(test_set.shape[0]):
                distance[j] = euclidean(p[j], test_set[j])

            self.kfold_error[i] = pd.Series(distance)
