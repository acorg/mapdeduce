"""BLUP: Best linear unbiased predictions of a linear mixed model."""

import numpy as np
import pandas as pd

from limix_legacy.modules.varianceDecomposition import VarianceDecomposition

from sklearn.model_selection import KFold

from scipy.spatial.distance import euclidean

from tqdm import tqdm

from .dataframes import SeqDf, CoordDf
from .hwas import cov
from .MapSeq import MapSeq


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


class FluLmmBlup(object):

    def __init__(self, filepath_or_df):
        """Train LMM and predict antigenic coordinates on influenza data.

        @param filepathor_df. Str. Filepath to csv file containing data to
            train the LMM. File should have four columns that correspond to:
                strain name, x coordinate, y coordinate, sequence
            Or: pd.DataFrame with the same columns
        """
        if type(filepath_or_df) is str:
            self.df = pd.read_csv(
                filepath_or_buffer=filepath_or_df,
                header=None,
                index_col=0,
                names=["strain", "x", "y", "seq"])
        else:
            self.df = filepath_or_df

        # Drop strains without a sequence or antigenic coordinates
        has_seq = self.df["seq"].notnull()
        has_coord = self.df.loc[:, ["x", "y"]].notnull().all(axis=1)
        self.df = self.df[has_seq & has_coord]

        # Expand sequences so that every amino acid is a different element in a
        # DataFrame
        self.seq = self.df["seq"].apply(lambda x: pd.Series(tuple(x)))
        self.seq.columns += 1
        self.seq = self.seq[range(1, 329)]

        # Coordinates
        self.coords = self.df[["x", "y"]]

    def predict(self, unknown_df):
        """Train a LMM using self.seq and self.coords as training data. Then
        predict coordinates of sequences in unknown_df.

        @param unknown_df. pd.DataFrame. Dataframe containing sequences with
            unknown antigenic coordinates. Must have columns names equal to
            range(1, 329).

        @returns Dataframe containing predicted antigenic coordinates.
        """
        # Concatenate the sequences to into one dataframe
        try:
            columns_match = (self.seq.columns == unknown_df.columns).all()

        except ValueError:
            raise ValueError("unknown_df does not have 328 columns")

        if not columns_match:
            raise ValueError(
                "unknown_df columns are not equal to range(1, 329)")

        seq_comb = pd.concat([self.seq, unknown_df])

        # Process sequences for running LMM
        seq_comb = SeqDf(seq_comb)
        seq_comb.remove_invariant(inplace=True)
        seq_comb.get_dummies(inplace=True)
        seq_comb.merge_duplicate_dummies(inplace=True)

        # Compute the covariance matrix of the dummies
        K = cov(seq_comb.dummies.values)

        # Make filler coordinates for the unknown sequences
        # Their value is irrelevant - they are not used in training
        n_unknown = unknown_df.shape[0]
        filler = np.zeros((n_unknown, 2))
        Y = np.vstack((self.coords.values, filler))

        blup = LmmBlup(Y=Y, K=K)

        n_known = self.df.shape[0]
        train = np.arange(n_known)
        test = np.arange(n_known, n_known + n_unknown)

        return pd.DataFrame(
            data=blup.predict(train=train, test=test),
            index=unknown_df.index)