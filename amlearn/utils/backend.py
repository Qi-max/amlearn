import os
import time
import pickle

import pandas as pd
import numpy as np
import yaml
from amlearn.utils.directory import auto_rename_file, create_path, write_file, \
    read_file, copy_path, delete_path
from amlearn.utils.logging import setup_logger, get_logger
from sklearn.externals import joblib

"""
All Amlearn naming conventions:
    If name start with "_", it's private function.
    If name end with "_", it's private property.
    
"""


class BackendContext(object):
    """Utility class to prepare amlearn needed paths.

    Args:
        output_path: str (default: /tmp/amlearn/task_%pid/output_%timestamp)
            amlearn beckend output path.
        tmp_path: str (default: /tmp/amlearn/task_%pid/tmp_%timestamp)
            Amlearn beckend temporary output path.
        delete_tmp_folder: boolean (default: True)
            Whether delete temporary output path after temporary persistence.
        auto_rename: boolean (default: True)
            Whether auto rename output path. (TODO: Now this features not done.)
        overwrite_path:
            Whether overwrite when output file exists.
        merge_path:
            Whether merge path when output file exists.
    """

    def __init__(self, output_path=None, tmp_path=None, delete_tmp_folder=True,
                 auto_rename=False, overwrite_path=False, merge_path=False):
        if output_path == tmp_path and (output_path is not None
                                        and output_path != 'tmp'
                                        and output_path != 'default'):
            raise ValueError('output_path should not be same with tmp_path.')

        self.delete_tmp_folder = delete_tmp_folder
        self.auto_rename = auto_rename
        self.overwrite_path = overwrite_path
        self.merge_path = merge_path
        class_name = self.__class__.__name__
        self._prepare_paths(output_path, tmp_path)
        logger_file = None if self.output_path_ is None else \
            os.path.join(self.output_path_, "amlearn_{}.log".format(class_name))
        setup_logger(logger_file=logger_file)
        self.logger_ = get_logger(class_name)
        if self.tmp_path is not None and self.output_path is not None :
            self.logger_.info("\n\t!!! Amlearn temporary output path is : {}\n"
                              "\t!!! Amlearn output path is : {}\n".format(
                                self.tmp_path, self.output_path))
        else:
            self.logger_.info("Haven't set beckend output path yet, the log will"
                              " only print on terminal.")

    @property
    def output_path(self):
        # Return the absolute path with ~ and environment variables expanded.
        if not hasattr(self, '_absoutput_path_'):
            self._absoutput_path_ = None if self.output_path_ is None else \
                os.path.expanduser(os.path.expandvars(self.output_path_))
        return self._absoutput_path_

    @property
    def tmp_path(self):
        # Return the absolute path with ~ and environment variables expanded.
        if not hasattr(self, 'abs_tmp_path_'):
            self.abs_tmp_path_ = None if self.tmp_path_ is None else \
                os.path.expanduser(os.path.expandvars(self.tmp_path_))
            if self.abs_tmp_path_ and self.abs_tmp_path_.endswith('/'):
                self.abs_tmp_path_ = self.abs_tmp_path_[:-1]
        return self.abs_tmp_path_

    def _prepare_paths(self, output_path=None, tmp_path=None,
                       auto_rename=False):
        timestamp = time.time()
        pid = os.getpid()

        if output_path == 'tmp' or output_path == 'default':
            output_path = \
                '/tmp/amlearn/task_%d/output_%d' % (pid, int(timestamp))
        self.output_path_ = output_path

        if tmp_path == 'tmp' or tmp_path == 'default':
            tmp_path = '/tmp/amlearn/task_%d/tmp_%d' % (pid, int(timestamp))
        self.tmp_path_ = tmp_path

        if output_path is not None:
            if auto_rename and os.path.exists(self.output_path_):
                self.output_path_ = auto_rename_file(self.output_path_)

            create_path(self.output_path_,
                        overwrite=self.overwrite_path, merge=self.merge_path)
            self.output_path_created_ = True

        if tmp_path is not None:
            if auto_rename and os.path.exists(self.tmp_path_):
                self.tmp_path_ = auto_rename_file(self.tmp_path_)

            create_path(self.tmp_path_,
                        overwrite=self.overwrite_path, merge=self.merge_path)
            self.tmp_path_created_ = True


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

        if self.tmp_path is not None:
            self.internals_path = os.path.join(self.tmp_path, ".amlearn")
            create_path(self.internals_path, merge=True)

    @property
    def output_path(self):
        return self.context.output_path

    @property
    def tmp_path(self):
        return self.context.tmp_path

    def tmp_persistence(self, tmp_path):
        check_path(self.tmp_path, 'tmp_path', 'BackendContext.tmp_path')
        if self.tmp_path != tmp_path:
            if tmp_path.startswith(self.tmp_path):
                sub_path = tmp_path[len(self.tmp_path) + 1:]
                output_path = os.path.join(self.output_path, sub_path)
                copy_path(tmp_path, output_path)
                if self.context.delete_tmp_folder:
                    delete_path(tmp_path)
            else:
                raise ValueError("{} should be Sub_folder of {}".format(
                    tmp_path, self.tmp_path))

    def _get_start_time_filename(self, seed):
        check_path(self.internals_path, 'internals_path',
                   'BackendContext.tmp_path')
        seed = int(seed)
        return os.path.join(self.internals_path, "start_time_%d" % seed)

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
    def _get_prediction_output_dir(self, name='all'):
        check_path(self.tmp_path, 'tmp_path',
                   'BackendContext.tmp_path')

        return os.path.join(self.tmp_path,
                            'predictions_{}'.format(name))

    @property
    def valid_predictions_type(self):
        if not hasattr(self, 'valid_predictions_type_'):
            functions = dir(self)
            self.valid_predictions_type_ = \
                [func.split('_')[-1] for func in functions
                 if func.startswith('save_predictions_as_')]
        return self.valid_predictions_type_

    def save_predictions_as_npy(self, predictions,
                                seed, name='all'):
        output_dir = self._get_prediction_output_dir(name)
        create_path(output_dir, merge=True)
        predict_file = os.path.join(output_dir,
                                    'predictions_{}_{}.npy'.format(name, seed))
        pickle.dump(predictions.astype(np.float32), predict_file)

    def save_predictions_as_txt(self, predictions,
                                seed, name='all'):
        output_dir = self._get_prediction_output_dir(name)
        create_path(output_dir, merge=True)
        predict_file = os.path.join(output_dir,
                                    'predictions_{}_{}.txt'.format(name, seed))

        with open(predict_file, 'w') as wf:
            wf.write("\n".join(list(map(str, predictions))))

    def save_predictions_as_dataframe(self, predictions,
                                      seed, name='all'):
        predict_dir = self._get_prediction_output_dir(name)
        create_path(predict_dir, merge=True)
        predict_file = os.path.join(predict_dir,
                                    'predictions_{}_{}.csv'.format(name, seed))

        predict_df = pd.DataFrame(predictions, columns=['predict'])
        predict_df.to_csv(predict_file)

    def _get_model_dir(self):
        check_path(self.tmp_path, 'tmp_path',
                   'BackendContext.tmp_path')

        return os.path.join(self.tmp_path)

    def save_model(self, model, seed):
        model_dir = self._get_model_dir()
        create_path(model_dir, merge=True)
        model_file = os.path.join(model_dir, 'model_{}.pkl'.format(seed))
        joblib.dump(model, model_file)

    def load_model(self, seed):
        model_dir = self._get_model_dir()
        model_file = os.path.join(model_dir, 'model_{}.pkl'.format(seed))
        model = joblib.load(model_file)
        return model

    @staticmethod
    def load_model_by_file(model_file):
        model = joblib.load(model_file)
        return model


class FeatureBackend(Backend):
    """Utility class to save featurized features."""
    def _get_featurizer_output_dir(self):
        check_path(self.tmp_path, 'tmp_path',
                   'BackendContext.tmp_path')

        return os.path.join(self.tmp_path, 'featurizer')

    def save_featurizer_as_dataframe(self, output_df, name='all'):
        featurizer_dir = self._get_featurizer_output_dir()
        create_path(featurizer_dir, merge=True)
        featurizer_file = os.path.join(featurizer_dir,
                                       'featurizer_{}.csv'.format(name))
        output_df.to_csv(featurizer_file)


def check_path(path, path_tag, setup_path_tag=None):
    if path is None:
        if setup_path_tag is None:
            setup_path_tag = path_tag
        raise EnvironmentError("{} is None, please set {} first!".format(
            path_tag, setup_path_tag))


def check_path_while_saving(path):
    if path is None:
        raise EnvironmentError(
            "If you want to save files, please set BackendContext's tmp_path "
            "and output_path, if you want to auto save files to "
            "'/tmp/amlearn/task_%timestamp/output_%pid', "
            "just set backend to 'tmp' or 'default', not None!")
