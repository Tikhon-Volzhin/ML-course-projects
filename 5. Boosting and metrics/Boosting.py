import contextlib
import inspect
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor
from scipy.special import softmax
from scipy.special import expit
from scipy.special import logit


class MyAdaBoostClassifier:
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            base_estimator: tp.Type[sklearn.base.BaseEstimator],
            seed: int,
            **kwargs
    ):
        self.n_classes = None
        self.error_history = [] 
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = kwargs
        signature = inspect.signature(self.base_estimator.__init__)
        self.seed_keyword = None
        if 'seed' in signature.parameters:
            self.seed_keyword = 'seed'
        elif 'random_state' in signature.parameters:
            self.seed_keyword = 'random_state'
        self.estimators = []


    def create_new_estimator(
            self,
            seed: int
    ):
        estimator = self.base_estimator(**dict(self.base_estimator_kwargs, **{self.seed_keyword : seed}))
        return estimator

    def get_new_weights(
            self,
            true_labels: np.ndarray,
            predictions: np.ndarray,
            weights: np.ndarray
    ):
        K, N = self.n_classes, predictions.shape[0]
        y = np.full((N, K), -1./(K - 1))
        np.put_along_axis(y, true_labels[:, None], 1., axis = 1)
        factor = -(K - 1) / K
        exp_factor = factor * np.diag(y @ np.log(np.clip(predictions.transpose(1, 0), self.eps, None)))
        new_weights = weights * np.exp(np.clip(exp_factor, None, self.big_number))
        return new_weights/ np.sum(new_weights)

    @staticmethod
    def get_estimator_error(
            estimator: sklearn.base.BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray
    ):
        return np.sum(weights * (estimator.predict(X) != y)) / np.sum(weights)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        self.error_history = []
        self.n_classes = len(np.unique(y))
        weights = np.full(X.shape[0], 1./X.shape[0])
        for seed in self.rng.choice(
                max(MyAdaBoostClassifier.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            self.estimators.append(self.create_new_estimator(seed))
            self.estimators[-1].fit(X, y, weights)
            predictions = self.estimators[-1].predict_proba(X)
            self.error_history.append(self.get_estimator_error(self.estimators[-1], X, y, weights))
            weights = self.get_new_weights(y, predictions, weights)

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        probas = np.zeros((X.shape[0], self.n_classes))
        for est in self.estimators:
            probas += est.predict_proba(X)
        probas /= self.n_estimators
        probas_softmax = softmax(probas, axis = 1)
        return probas_softmax

    def predict(
            self,
            X: np.ndarray
    ):
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis = 1)
        return predictions


class MyBinaryTreeGradientBoostingClassifier: 
    big_number = 1 << 32 
    eps = 1e-8 
 
    def __init__( 
            self, 
            n_estimators: int, 
            learning_rate: float, 
            seed: int, 
            **kwargs 
    ): 
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.initial_logits = None 
        self.rng = np.random.default_rng(seed) 
        self.base_estimator = DecisionTreeRegressor 
        self.base_estimator_kwargs = kwargs 
        self.estimators = [] 
        self.loss_history = []
 
    def create_new_estimator(self, seed): 
        return self.base_estimator(**self.base_estimator_kwargs, random_state=seed)

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        eps = 1e-8
        prob = np.clip(expit(logits), eps, 1. - eps)
        nll = - (true_labels * np.log(prob) + (1 - true_labels) * np.log(1. - prob)).sum()
        return nll

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        eps = 1e-8
        prob = np.clip(expit(logits), eps, 1. - eps)
        gradient = - (true_labels - prob)
        return gradient

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        self.loss_history = []
        assert (np.unique(y) == np.arange(2)).all()
        self.initial_logits = logit(np.mean(y))
        logits = np.full(X.shape[0], self.initial_logits)
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            self.estimators.append(self.create_new_estimator(seed))
            gradient = self.cross_entropy_loss_gradient(y, logits)
            self.estimators[-1].fit(X, gradient)
            logits -= self.learning_rate* self.estimators[-1].predict(X)
            self.loss_history.append(self.cross_entropy_loss(y, logits))

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        logits =  np.full(X.shape[0], self.initial_logits)
        for estimator in self.estimators:
            logits -= self.learning_rate * estimator.predict(X)
        return expit(logits)

    def predict(
            self,
            X: np.ndarray
    ):
        proba = self.predict_proba(X)
        proba[proba >= 0.5] = 1.
        proba[proba < 0.5] = 0.
        return proba




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
        # elif name in self.leaderboard.index:
        #     raise NameError("Run with given name already exists, name should be unique")
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
            cv_iterable: tp.Iterable,
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
        cross_val_results = sklearn.model_selection.cross_validate(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            scoring=metrics
        )
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
        cross_val_stacking_prediction = sklearn.model_selection.cross_val_predict(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            method=method
        )
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
