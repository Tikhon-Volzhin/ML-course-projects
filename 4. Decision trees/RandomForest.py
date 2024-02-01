import contextlib
import enum
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import mode
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

class NodeType(enum.Enum):
    REGULAR = 1
    TERMINAL = 2

def find_best_split_numpy(feature_vector, target_vector):
    sort_idx = np.argsort(feature_vector)
    feature_vector, target_vector = feature_vector[sort_idx], target_vector[sort_idx]
    unique_feature_vector, indx, count = np.unique(feature_vector, return_index=True, return_counts=True)
    if len(unique_feature_vector) == 1:
        return None, None, None, None
    R_len = len(target_vector)
    left_range, right_range = np.arange(1, R_len + 1)[indx + count - 1], np.arange(1, R_len + 1)[::-1][indx]
    ones_imp_left = (np.cumsum(target_vector)[indx + count - 1]/left_range)[:-1]
    ones_imp_right = (np.cumsum(target_vector[::-1])[::-1][indx]/right_range)[1:]
    ginis_val = -2/ R_len * ((1 - ones_imp_left) * ones_imp_left * left_range[:-1] + (1 - ones_imp_right) * ones_imp_right * right_range[1:])
    thresholds = unique_feature_vector[:-1] + np.diff(unique_feature_vector)/2
    best_gini_idx = np.argmax(ginis_val)
    return thresholds, ginis_val, thresholds[best_gini_idx], ginis_val[best_gini_idx]

def init_gini(targets):
    ones_imp = (targets == 1).sum()/len(targets)
    zeros_imp = 1 - ones_imp
    return -(1 - ones_imp * ones_imp - zeros_imp * zeros_imp)


def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts = True)
    gini_index = 0
    len_y = len(y)
    for count in counts:
        p_k = count/len_y
        gini_index+= p_k * (1 - p_k)
    return gini_index


def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> \
        tp.Tuple[float, float, float]:
    full_len = len(y_left) + len(y_right)
    left_impurity =  gini(y_left)
    right_impurity =  gini(y_right)
    weighted_impurity = len(y_left)/full_len * left_impurity + len(y_right)/full_len * right_impurity
    return weighted_impurity, left_impurity, right_impurity


def create_split(feature_values: np.ndarray, threshold: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    left_idx = (feature_values <= threshold).nonzero()[0]
    right_idx = (feature_values > threshold).nonzero()[0]
    return left_idx, right_idx

def _best_split(self, X: np.ndarray, y: np.ndarray):
    lowest_impurity = np.inf
    best_feature_id = None
    best_threshold = None
    lowest_left_child_impurity, lowest_right_child_impurity = None, None
    features = self._meta.rng.permutation(X.shape[1])
    for feature in features:
        current_feature_values = X[:, feature]
        thresholds = np.unique(current_feature_values)
        for threshold in thresholds:
            left_idx, right_idx = create_split(current_feature_values, threshold)
            current_weighted_impurity, current_left_impurity, current_right_impurity = weighted_impurity(y[left_idx], y[right_idx])
            if current_weighted_impurity < lowest_impurity:
                lowest_impurity = current_weighted_impurity
                best_feature_id = feature
                best_threshold = threshold
                lowest_left_child_impurity = current_left_impurity
                lowest_right_child_impurity = current_right_impurity

    return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity

class MyDecisionTreeNode:
    def __init__(
            self,
            meta: 'MyDecisionTreeClassifier',
            depth,
            node_type: NodeType = NodeType.REGULAR,
            predicted_class: tp.Optional[tp.Union[int, str]] = None,
            left_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            right_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            feature_id: int = None,
            threshold: float = None,
            impurity: float = np.inf
    ):
        self._node_type = node_type
        self._meta = meta
        self._depth = depth
        self._predicted_class = predicted_class
        self._class_proba = None
        self._left_subtree = left_subtree
        self._right_subtree = right_subtree
        self._feature_id = feature_id
        self._threshold = threshold
        self._impurity = impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        lowest_impurity = np.inf
        best_feature_id = None
        best_threshold = None
        lowest_left_child_impurity, lowest_right_child_impurity = None, None
        features = self._meta.rng.permutation(X.shape[1])
        for feature in features:
            current_feature_values = X[:, feature]
            thresholds = np.unique(current_feature_values)
            for threshold in thresholds:
                left_idx, right_idx = create_split(current_feature_values, threshold)
                current_weighted_impurity, current_left_impurity, current_right_impurity = weighted_impurity(y[left_idx], y[right_idx])
                if current_weighted_impurity < lowest_impurity:
                    lowest_impurity = current_weighted_impurity
                    best_feature_id = feature
                    best_threshold = threshold
                    lowest_left_child_impurity = current_left_impurity
                    lowest_right_child_impurity = current_right_impurity

        return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity

    def fit(self, X: np.ndarray, y: np.ndarray):
        uniq, count = np.unique(y, return_counts=True)
        if (len(uniq) == 1 or self._meta.max_depth <= self._depth or self._meta.min_samples_split > len(y)):
            self._node_type = NodeType.TERMINAL
            self._predicted_class = uniq[np.argmax(count)]
            proba_ndarray = np.zeros(self._meta._n_classes)
            proba_ndarray[uniq] = count/len(y)
            self._class_proba = proba_ndarray
            return self

        self._feature_id, self._threshold, left_imp, right_imp = self._best_split(X, y)
        left_idx, right_idx = create_split(X[:, self._feature_id], self._threshold)
        self._left_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1,  
            impurity=left_imp
        ).fit(X[left_idx], y[left_idx])
        self._right_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1, 
            impurity=right_imp
        ).fit(X[right_idx], y[right_idx])
        return self

    def predict(self, x: np.ndarray):
        if self._node_type is NodeType.TERMINAL:
            return self._predicted_class
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict(x)
        else:
            return self._right_subtree.predict(x)

    def predict_proba(self, x: np.ndarray):
        if self._node_type is NodeType.TERMINAL:
            return self._class_proba
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict_proba(x)
        else:
            return self._right_subtree.predict_proba(x)



class MyDecisionTreeClassifier:

    def __init__(
            self,
            max_depth: tp.Optional[int] = None,
            min_samples_split: tp.Optional[int] = 2,
            seed: int = 0
    ):
        self.root = MyDecisionTreeNode(self, 1)
        self._is_trained = False
        self.max_depth = max_depth or np.inf
        self.min_samples_split = min_samples_split or 2
        self.rng = np.random.default_rng(seed)
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = np.unique(y).shape[0]
        self.root.fit(X, y)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError('untrained model')
        else:
            result = np.empty(X.shape[0])
            for i, x in enumerate(X):
                result[i] = self.root.predict(x)
            return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError('untrained model')
        else:
            result = np.empty((X.shape[0], self._n_classes))
            for i, x in enumerate(X):
                result[i] = self.root.predict_proba(x)
            return result


class MyRandomForestClassifier:
    big_number = 1 << 32

    def __init__(
            self,
            n_estimators: int,
            max_depth: tp.Optional[int] = None,
            min_samples_split: tp.Optional[int] = 2,
            seed: int = 0
    ):
        self._n_classes = 0
        self._is_trained = False
        self.rng = np.random.default_rng(seed)
        self.estimators = [
            MyDecisionTreeClassifier(max_depth, min_samples_split, seed=seed) for
            seed in self.rng.choice(max(MyRandomForestClassifier.big_number, n_estimators), size=(n_estimators,),
                                    replace=False)]

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        indices = self.rng.choice(len(y), size= len(y))
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = np.unique(y).shape[0]
        for estimator in self.estimators:
            X_, y_ = self._bootstrap_sample(X, y)
            estimator.fit(X_, y_)
        self._is_trained = True
        return self

    def predict_proba(self, X: np.ndarray):
        probas = np.zeros((X.shape[0], self._n_classes))
        for estimator in self.estimators:
            probas += estimator.predict_proba(X)
        probas /= len(self.estimators)
        return probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)





class Logger:
    """Logger performs data management and stores scores and other relevant information"""

    def __init__(self, logs_path: tp.Union[str, os.PathLike]):
        self.path = pathlib.Path(logs_path)

        records = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.json'):
                    uuid = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            logged_data = json.load(f)
                            records.append(
                                {
                                    'id': uuid,
                                    **logged_data
                                }
                            )
                        except json.JSONDecodeError:
                            pass
        if records:
            self.leaderboard = pd.DataFrame.from_records(records, index='id')
        else:
            self.leaderboard = pd.DataFrame(index=pd.Index([], name='id'))

        self._current_run = None

    class Run:
        """Run incapsulates information for a particular entry of logged material. Each run is solitary experiment"""

        def __init__(self, name, storage, path):
            self.name = name
            self._storage = storage
            self._path = path
            self._storage.append(pd.Series(name=name))

        def log(self, key, value):
            self._storage.loc[self.name, key] = value

        def log_values(self, log_values: tp.Dict[str, tp.Any]):
            for key, value in log_values.items():
                self.log(key, value)

        def save_logs(self):
            with open(self._path / f'{self.name}.json', 'w+') as f:
                json.dump(self._storage.loc[self.name].to_dict(), f)

        def log_artifact(self, fname: str, writer: tp.Callable):
            with open(self._path / fname, 'wb+') as f:
                writer(f)

    @contextlib.contextmanager
    def run(self, name: tp.Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        elif name in self.leaderboard.index:
            raise NameError("Run with given name already exists, name should be unique")
        else:
            name = name.replace(' ', '_')
        self._current_run = Logger.Run(name, self.leaderboard, self.path / name)
        os.makedirs(self.path / name, exist_ok=True)
        try:
            yield self._current_run
        finally:
            self._current_run.save_logs()


def load_predictions_dataframe(filename, column_prefix, index):
    with open(filename, 'rb') as file:
        data = np.load(file)
        dataframe = pd.DataFrame(data, columns=[f'{column_prefix}_{i}' for i in range(data.shape[1])],
                                 index=index)
        return dataframe


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'

    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            cv_iterable: tp.Union[sklearn.model_selection.KFold, tp.Iterable],
            logger: Logger,
            metrics: tp.Dict[str, tp.Union[tp.Callable, str]],
            n_jobs=-1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._cv_iterable = cv_iterable
        self.logger = logger
        self._metrics = metrics
        self._n_jobs = n_jobs

    def score_test(self, estimator, metrics, run, test_data=None):
        """
        Computes scores for test data and logs them to given run
        :param estimator: fitted estimator
        :param metrics: metrics to compute
        :param run: run to log into
        :param test_data: optional argument if one wants to pass augmented test dataset
        :return: None
        """
        if test_data is None:
            test_data = self.X_test
        test_scores = _score(estimator, test_data, self.y_test, metrics)
        run.log_values({key + '_test': value for key, value in test_scores.items()})

    def score_cv(self, estimator, metrics, run):
        """
        computes scores on cross-validation
        :param estimator: estimator to fit
        :param metrics: metrics to compute
        :param run: run to log to
        :return: None
        """
        cross_val_results = cross_val_score(estimator, self.X_train, self.y_train,  cv = self._cv_iterable, scoring = metrics) 
        for key, value in cross_val_results.items():
            if key.startswith('test_'):
                metric_name = key.split('_', maxsplit=1)[1]
                mean_score = np.mean(value)
                std_score = np.std(value)
                run.log_values(
                    {
                        metric_name + '_mean': mean_score,
                        metric_name + '_std': std_score
                    }
                )

    def generate_stacking_predictions(self, estimator, run):
        """
        generates predictions over cross-validation folds, then saves them as artifacts
        returns fitted estimator for convinience and les train overhead
        :param estimator: estimator to use
        :param run: run to log to
        :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
        """
        if hasattr(estimator, "predict_proba"):
            method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"
        cross_val_stacking_prediction = cross_val_predict(estimator, self.X_train, self.y_train,  cv = self._cv_iterable, method = method) 
        run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                         lambda file: np.save(file, cross_val_stacking_prediction))
        estimator.fit(self.X_train, self.y_train)
        test_stacking_prediction = getattr(estimator, method)(self.X_test)
        run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                         lambda file: np.save(file, test_stacking_prediction))
        return estimator, cross_val_stacking_prediction, test_stacking_prediction

    def get_metrics(self, estimator):
        """
        get callable metrics with estimator validation
        (e.g. estimator has predict_proba necessary for likelihood computation, etc)
        """
        return _check_multimetric_scoring(estimator, self._metrics)

    def run(self, estimator: sklearn.base.BaseEstimator, name=None):
        """
        perform run for given estimator
        :param estimator: estimator to use
        :param name: name of run for convinience and consitent logging
        :return: leaderboard with conducted run
        """
        metrics = self.get_metrics(estimator)
        with self.logger.run(name=name) as run:
            # compute predictions over cross-validation
            self.score_cv(estimator, metrics, run)
            fitted_on_train, _, _ = self.generate_stacking_predictions(estimator, run)
            self.score_test(fitted_on_train, metrics, run, test_data=self.X_test)
            return self.logger.leaderboard.loc[[run.name]]

    def get_stacking_predictions(self, run_names):
        """
        :param run_names: run names for which to extract stacking predictions for averaging and stacking
        :return: dataframe with predictions indexed by run names
        """
        train_dataframes = []
        test_dataframes = []
        for run_name in run_names:
            train_filename = self.logger.path / run_name / ExperimentHandler.stacking_prediction_filename
            train_dataframes.append(load_predictions_dataframe(filename=train_filename, column_prefix=run_name,
                                                               index=self.X_train.index))
            test_filename = self.logger.path / run_name / ExperimentHandler.test_stacking_prediction_filename
            test_dataframes.append(load_predictions_dataframe(filename=test_filename, column_prefix=run_name,
                                                              index=self.X_test.index))

        return pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)


def score_cv(self, estimator, metrics, run):
        """
        computes scores on cross-validation
        :param estimator: estimator to fit
        :param metrics: metrics to compute
        :param run: run to log to
        :return: None
        """
        cross_val_results = cross_val_score(estimator, self.X_train, self.y_train,  cv = self._cv_iterable, scoring = metrics)
        for key, value in cross_val_results.items():
            if key.startswith('test_'):
                metric_name = key.split('_', maxsplit=1)[1]
                mean_score = np.mean(value)
                std_score = np.std(value)
                run.log_values(
                    {
                        metric_name + '_mean': mean_score,
                        metric_name + '_std': std_score
                    }
                )

def generate_stacking_predictions(self, estimator, run):
    """
    generates predictions over cross-validation folds, then saves them as artifacts
    returns fitted estimator for convinience and les train overhead
    :param estimator: estimator to use
    :param run: run to log to
    :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
    """
    if hasattr(estimator, "predict_proba"):
        method = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        method = "decision_function"
    else:
        method = "predict"
    cross_val_stacking_prediction = cross_val_predict(estimator, self.X_train, self.y_train,  cv = self._cv_iterable, method = method) 
    run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                        lambda file: np.save(file, cross_val_stacking_prediction))
    estimator.fit(self.X_train, self.y_train)
    test_stacking_prediction = getattr(estimator, method)(self.X_test)
    run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                        lambda file: np.save(file, test_stacking_prediction))
    return estimator, cross_val_stacking_prediction, test_stacking_prediction