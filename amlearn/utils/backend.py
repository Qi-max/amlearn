import os
import time
import pickle

import pandas as pd
import numpy as np
from amlearn.utils.directory import auto_rename_file, create_path, write_file, \
    read_file
from amlearn.utils.logging import setup_logger, get_logger
from sklearn.externals import joblib


class BackendContext(object):
    """Utility class to prepare amlearn needed paths.

    Args:
        output_path:
        temp_path:
        delete_tmp_folder:
        auto_rename:
        overwrite_path:
        merge_path:
    """

    def __init__(self, output_path, temp_path=None, delete_tmp_folder=True,
                 auto_rename=False, overwrite_path=False, merge_path=False):
        if output_path == temp_path and output_path is not None:
            raise ValueError('output_path should not be same with temp_path.')

        self.delete_tmp_folder = delete_tmp_folder
        self.auto_rename = auto_rename
        self.overwrite_path = overwrite_path
        self.merge_path = merge_path
        self._prepare_paths(output_path, temp_path)
        class_name = self.__class__.__name__
        setup_logger(os.path.join(self._output_path,
                                  "amlearn_{}.log".format(class_name)))
        self._logger = get_logger(class_name)

    @property
    def output_path(self):
        # Return the absolute path with ~ and environment variables expanded.
        return os.path.expanduser(os.path.expandvars(self._output_path))

    @property
    def temp_path(self):
        # Return the absolute path with ~ and environment variables expanded.
        return os.path.expanduser(os.path.expandvars(self._temp_path))

    def _prepare_paths(self, output_path, temp_path=None,
                       auto_rename=False, overwrite=False):
        timestamp = time.time()
        pid = os.getpid()

        self._output_path = output_path if output_path \
            else '/tmp/amlearn/task_%d/output_%d' % (pid, int(timestamp))

        self._temp_path = temp_path if temp_path \
            else '/tmp/amlearn/task_%d/tmp_%d' % (pid, int(timestamp))

        if auto_rename:
            if os.path.exists(self._output_path):
                self._output_path = auto_rename_file(self._output_path)
            if os.path.exists(self._temp_path):
                self._temp_path = auto_rename_file(self._temp_path)

        create_path(self._output_path,
                    overwrite=self.overwrite_path, merge=self.merge_path)
        self._output_path_created = True

        create_path(self._temp_path,
                    overwrite=self.overwrite_path, merge=self.merge_path)
        self._temp_path_created = True


class MLBackend(object):
    def __init__(self, context):
        self.context = context
        if not os.path.exists(self.output_path):
            raise ValueError(
                "Output path {} does not exist.".format(self.output_path))
        self.logger = get_logger(self.__class__.__name__)
        self.internals_path = os.path.join(self.temp_path, ".amlearn")
        create_path(self.internals_path, merge=True)

    @property
    def output_path(self):
        return self.context.output_path

    @property
    def temp_path(self):
        return self.context.temp_path

    def _get_start_time_filename(self, seed):
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

    def _get_prediction_output_dir(self, name='all'):
        return os.path.join(self.output_path,
                            'predictions_{}'.format(name))

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
        return os.path.join(self.output_path)

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

