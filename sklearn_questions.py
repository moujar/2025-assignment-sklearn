"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        # checks if data is continuous or not
        # this means if classes are weird (like -1s or floats or whatever)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)  # not sure if needed

        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        distances = pairwise_distances(X, self.X_)  # reminder, X_ is TRAINING

        if self.n_neighbors == 1:
            # for each x test (axis=1), get the MINIMUM distance index
            # change result from ([indexes],) to ([indexes],1) (for example),
            # basically, it needs to match the SIZE that the
            # rest of the code is expecting.
            neighbor_indices = np.argmin(distances, axis=1)[:, np.newaxis]
        else:
            # argpartition -> orders around the
            # kth element (smaller first, larger after)
            # and then continue as above.
            neighbor_indices = np.argpartition(
                distances,
                self.n_neighbors,
                axis=1)[:, :self.n_neighbors]
        # look at the matches. eg. look at ur
        # closest X's classes. take majority

        neighbor_labels = self.y_[neighbor_indices]

        for sample_idx in range(X.shape[0]):  # goes through all x tests
            unique_classes, class_counts = np.unique(
                neighbor_labels[sample_idx],
                return_counts=True)
            y_pred[sample_idx] = unique_classes[np.argmax(class_counts)]

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        # score means accuracy -> average
        y_pred = self.predict(X)

        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if self.time_col == "index":  # check if time_col is set to index
            if not hasattr(X, "index"):  # throw an error if not
                raise ValueError("X must have a datetime index")
            # set datetime_values to the datetimes in index
            datetime_values = X.index
        else:
            if not isinstance(X, pd.DataFrame):  # is it a df?
                raise ValueError(
                    "X must be a DataFrame with a datetime column"
                )
            # if a df and index wasn't our time_col,
            # use whatever our time_col is
            datetime_values = X[self.time_col]

        # ensure all datetime_values are actually datetimes
        datetime_values = pd.to_datetime(datetime_values, errors="raise")
        if isinstance(datetime_values, pd.Series):
            # sort them !
            month_periods = datetime_values.dt.to_period('M').sort_values()
        else:
            # sort them !
            month_periods = datetime_values.to_period('M').sort_values()

        # if only one month, then 0 splits, otherwise its total - 1 splits
        return max(0, month_periods.nunique() - 1)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        if self.time_col == "index":
            if not hasattr(X, "index"):
                raise ValueError("X must have a datetime index")
            datetime_values = X.index
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "X must be a DataFrame with a datetime column")
            datetime_values = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(datetime_values):
            raise ValueError("Not a datetime")

        datetime_values = pd.to_datetime(datetime_values, errors="raise")
        if isinstance(datetime_values, pd.Series):
            month_periods = datetime_values.dt.to_period("M")
        else:
            month_periods = datetime_values.to_period("M")

        # figure out the sort order
        sort_indices = np.argsort(datetime_values.values)
        # Use iloc for Series, direct indexing for Index
        if isinstance(month_periods, pd.Series):
            sorted_month_periods = month_periods.iloc[sort_indices]
        else:
            sorted_month_periods = month_periods[sort_indices]

        # get the unique months
        unique_month_periods = sorted_month_periods.unique()
        n_splits = self.get_n_splits(X, y, groups)  # get the number of splits

        # each month will be a test month
        for split_idx in range(n_splits):
            train_month = unique_month_periods[split_idx]
            test_month = unique_month_periods[split_idx + 1]

            # train on the month before test month
            train_month_mask = sorted_month_periods == train_month
            # test on the next month
            test_month_mask = sorted_month_periods == test_month
            # sort_indices[train_month_mask] gives original DataFrame
            # positions where the month matches train_month
            idx_train = sort_indices[train_month_mask]
            idx_test = sort_indices[test_month_mask]

            yield idx_train, idx_test
