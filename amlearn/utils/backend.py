import os
import json
import time
import yaml
import pickle
import joblib
import pandas as pd
import numpy as np
from amlearn.utils.directory import auto_rename_file, create_path, \
    write_file, read_file
from amlearn.utils.logging import setup_logger, get_logger

"""
All Amlearn naming conventions:
    If name start with "_", it's private function.
    If name end with "_", it's private property.
    
"""

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class BackendContext(object):
    """Utility class to prepare amlearn needed paths.

    Args:
        output_path: str (default: /tmp/amlearn/task_%pid/output_%timestamp)
            amlearn beckend output path.
        delete_tmp_folder: boolean (default: True)
            Whether delete temporary output path after temporary persistence.
        auto_rename: boolean (default: True)
            Whether auto rename output path. (TODO: Now this features not done.)
        overwrite_path:
            Whether overwrite when output file exists.
        merge_path:
            Whether merge path when output file exists.
    """

    def __init__(self, output_path=None, auto_rename=False,
                 overwrite_path=False, merge_path=False):
        self.auto_rename = auto_rename
        self.overwrite_path = overwrite_path
        self.merge_path = merge_path
        class_name = self.__class__.__name__
        self._prepare_paths(output_path)
        logger_file = None if self.output_path_ is None else \
            os.path.join(self.output_path_, "amlearn_{}.log".format(class_name))
        setup_logger(logger_file=logger_file)
        self.logger_ = get_logger(class_name)
        if self.output_path is not None :
            self.logger_.info("\n\t!!! Amlearn output path is : {}\n".format(
                self.output_path))
        else:
            self.logger_.info("Haven't set beckend output path yet, the log "
                              "will only print on terminal.")

    @property
    def output_path(self):
        # Return the absolute path with ~ and environment variables expanded.
        if not hasattr(self, '_absoutput_path_'):
            self._absoutput_path_ = None if self.output_path_ is None else \
                os.path.expanduser(os.path.expandvars(self.output_path_))
        return self._absoutput_path_


    def _prepare_paths(self, output_path=None, auto_rename=False):
        timestamp = time.time()
        pid = os.getpid()

        if output_path == 'tmp' or output_path == 'default':
            output_path = \
                '/tmp/amlearn/task_%d/output_%d' % (pid, int(timestamp))
        self.output_path_ = output_path

        if output_path is not None:
            if auto_rename and os.path.exists(self.output_path_):
                self.output_path_ = auto_rename_file(self.output_path_)

            create_path(self.output_path_,
                        overwrite=self.overwrite_path, merge=self.merge_path)
            self.output_path_created_ = True


class Backend(object):
    """Utility class to load default environment, persistent output path,
       calculate running time, and so on.
    """
    def __init__(self, context):
        self.context = context
        if self.output_path and not os.path.exists(self.output_path):
            raise ValueError(
                "Output path {} does not exist.".format(self.output_path))

        self.logger = get_logger(self.__class__.__name__)

    @property
    def output_path(self):
        return self.context.output_path

    def _get_start_time_filename(self, seed):
        check_path(self.output_path, 'start_time_path',
                   'BackendContext.output_path')
        seed = int(seed)
        return os.path.join(self.output_path, "start_time_%d" % seed)

    def save_start_time(self, seed):
        start_time = time.time()
        time_file = self._get_start_time_filename(seed)
        write_file(time_file, str(start_time))
        return time_file

    def load_start_time(self, seed):
        start_time = float(read_file(self._get_start_time_filename(seed))[0])
        return start_time

    @property
    def def_env(self):
        if not hasattr(self, "def_env_"):
            with open(os.path.join(os.path.dirname(__file__),
                                   'default_environment.yaml'), 'r') as lf:
                self.def_env_ = yaml.load(lf)
        return self.def_env_


class MLBackend(Backend):
    """Utility class to load model, save model and save predictions."""
    def _get_prediction_output_dir(self, sub_dir='prediction'):
        check_path(self.output_path, 'output_path',
                   'BackendContext.output_path')
        return os.path.join(self.output_path, sub_dir)

    @property
    def valid_predictions_type(self):
        if not hasattr(self, 'valid_predictions_type_'):
            functions = dir(self)
            self.valid_predictions_type_ = \
                [func.split('_')[-1] for func in functions
                 if func.startswith('save_predictions_as_')]
        return self.valid_predictions_type_

    def save_predictions_as_npy(self, predictions, sub_dir='prediction',
                                name='prediction', seed=None):
        output_dir = self._get_prediction_output_dir(sub_dir)
        create_path(output_dir, merge=True)
        predict_file = os.path.join(output_dir,
                                    '{}.npy'.format(name) if seed is None
                                    else '{}_{}.npy'.format(name, seed))
        np.save(predict_file, predictions)

    def save_predictions_as_pickle(self, predictions, sub_dir='prediction',
                                name='prediction', seed=None):
        output_dir = self._get_prediction_output_dir(sub_dir)
        create_path(output_dir, merge=True)
        predict_file = os.path.join(output_dir,
                                    '{}.pickle'.format(name) if seed is None
                                    else '{}_{}.pickle'.format(name, seed))
        pickle.dump(predictions.astype(np.float32), predict_file)

    def save_predictions_as_txt(self, predictions, sub_dir='prediction',
                                name='prediction', seed=None):
        output_dir = self._get_prediction_output_dir(sub_dir)
        create_path(output_dir, merge=True)
        predict_file = os.path.join(output_dir,
                                    '{}.txt'.format(name) if seed is None
                                    else '{}_{}.txt'.format(name, seed))

        with open(predict_file, 'w') as wf:
            wf.write("\n".join(list(map(str, predictions))))

    def save_predictions_as_dataframe(self, predictions, subdir='prediction',
                                      name='prediction', seed=None):
        predict_dir = self._get_prediction_output_dir(subdir)
        create_path(predict_dir, merge=True)
        predict_file = os.path.join(predict_dir,
                                    '{}.csv'.format(name) if seed is None
                                    else '{}_{}.csv'.format(name, seed))

        predict_df = pd.DataFrame(predictions,
                                  columns=['target', 'predict']
                                  if isinstance(predictions[0], np.ndarray)
                                  else ['predict'])
        predict_df.to_csv(predict_file)

    def save_json(self, data, sub_dir='json', name='json_file', seed=None):
        json_dir = self._get_dir(sub_dir)
        create_path(json_dir, merge=True)
        json_file = os.path.join(json_dir,
                                  '{}.json'.format(name) if seed is None
                                  else '{}_{}.json'.format(name, seed))
        with open(json_file, 'w') as wf:
            json.dump(data, wf)

    def _get_dir(self, sub_dir='model'):
        check_path(self.output_path, 'output_path',
                   'BackendContext.output_path')

        return os.path.join(self.output_path, sub_dir)

    def save_model(self, model, sub_dir='model', name='model', seed=None):
        model_dir = self._get_dir(sub_dir)
        create_path(model_dir, merge=True)
        model_file = os.path.join(model_dir,
                                  '{}.pkl'.format(name) if seed is None
                                  else '{}_{}.pkl'.format(name, seed))
        joblib.dump(model, model_file)

    def load_model(self, sub_dir='model', name='model', seed=None):
        model_dir = self._get_dir(sub_dir)
        model_file = os.path.join(model_dir,
                                  '{}.pkl'.format(name) if seed is None
                                  else '{}_{}.pkl'.format(name, seed))
        model = joblib.load(model_file)
        return model

    @staticmethod
    def load_model_by_file(model_file):
        model = joblib.load(model_file)
        return model


class FeatureBackend(Backend):
    """Utility class to save featurized features."""
    def _get_featurizer_output_dir(self):
        check_path(self.output_path, 'output_path',
                   'BackendContext.output_path')
        return os.path.join(self.output_path, 'featurizer')

    def save_featurizer_as_dataframe(self, output_df, name='featurizer',
                                     save_type='pickle.gz'):
        featurizer_dir = self._get_featurizer_output_dir()
        create_path(featurizer_dir, merge=True)
        featurizer_file = os.path.join(
            featurizer_dir, '{}.{}'.format(name, save_type))
        if save_type == 'csv':
            output_df.to_csv(featurizer_file)
        elif save_type.startswith('pickle'):
            output_df.to_pickle(featurizer_file)


def check_path(path, path_tag, setup_path_tag=None):
    if path is None:
        if setup_path_tag is None:
            setup_path_tag = path_tag
        raise EnvironmentError("{} is None, please set {} first!".format(
            path_tag, setup_path_tag))


def check_path_while_saving(path):
    if path is None:
        raise EnvironmentError(
            "If you want to save files, please set BackendContext's "
            "output_path, if you want to auto save files to "
            "'/tmp/amlearn/task_%timestamp/output_%pid', "
            "just set backend to 'tmp' or 'default', not None!")
