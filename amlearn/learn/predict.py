import os
import joblib
import numpy as np
from amlearn.learn.base import create_ml_backend
from amlearn.learn.sklearn_patch import calc_scores
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file
from sklearn.base import ClassifierMixin, RegressorMixin

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


def load_model_and_predict(model_file, X, y=None,
                           task="classification", scoring=None,
                           save_prediction_types='dataframe',
                           backend=None, output_path='tmp'):
    """
    Predict on an arbitrary dataset using the trained model and save predictions
    and/or scores.

    Args:
        model_file: str
            Path to the trained model file.
        X: array-like
            The data to fit. Can be for example a list, or an array.
        y: array-like, optional (default: None)
            The target variable to predict in the case of supervise learning.
        task: str, 'classification' or 'regression' (default: classification)
            Model's task type, only support 'classification' or 'regression'
        scoring: str or callable or a list of them or None (default: None)
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
        save_prediction_types: str or [str] (default: dataframe)
            It effect when save_prediction is True.
            The optional parameters are: ["npy", "txt", "dataframe"].
        backend: Backend object (default: None)
            MLBackend object which defined output_path, environment
            configuration, save_predictions, and so on.
            If None, use default MLBackend object.
        output_path: str (default: 'tmp')
            Output path of PredictPipeline, if is None or 'tmp', use the
            default output path: '/tmp/amlearn/task_%pid/output_%timestamp'.

    Returns:
        predictions: np.array
            Predictions from the trained model.
    """
    if backend is None:
        backend = create_ml_backend(output_path=output_path)

    model = joblib.load(model_file)
    # print(model)
    if isinstance(model, RegressorMixin):
        if task == 'regression' or task is None:
            task = 'regression'
            if scoring is None:
                scoring = ['r2', 'neg_mean_absolute_error',
                           'neg_mean_squared_error']
        else:
            raise TypeError('Model type of model_file is "regression", '
                            'but the task parameter is not. Please make '
                            'sure these two match.')
    elif isinstance(model, ClassifierMixin):
        if task == 'classification' or task is None:
            task = 'classification'
            if scoring is None:
                scoring = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']
        else:
            raise TypeError('Model type of model_file is "classification",'
                            'but the task parameter is not. Please make '
                            'sure these two match.')
    else:
        raise TypeError('Model must be instance of RegressorMixin or '
                        'ClassifierMixin.')

    if task == 'classification':
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            predictions = model.decision_function(X)
        else:
            predictions = model.predict(X)

        targets_and_predictions = np.array(list(zip(y, predictions[:, 1]))) \
            if y is not None else predictions[:, 1]

    elif task == 'regression':
        predictions = model.predict(X)
        targets_and_predictions = np.array(list(zip(y, predictions))) \
            if y is not None else predictions
    else:
        raise ValueError('task only support classification or regression')

    if scoring and y is not None:
        scores, _ = calc_scores(X=X, y=y, estimator=model, scoring=scoring)
        write_file(os.path.join(backend.output_path, 'scores.txt'),
                   '{}\n{}'.format(
                       ','.join(['dataset'] + list(scores.keys())),
                       ','.join(['predict'] + list(map(str, scores.values())))))

    if not isinstance(save_prediction_types, list_like()):
        save_prediction_types = [save_prediction_types]
    for predict_type in save_prediction_types:
        if predict_type in backend.valid_predictions_type:
            getattr(backend, 'save_predictions_as_{}'.format(predict_type))\
                (targets_and_predictions, subdir='')
        else:
            raise ValueError('predict_type {} is unknown, '
                             'Possible values are {}'.format(
                    predict_type, backend.valid_predictions_type))

    return predictions
