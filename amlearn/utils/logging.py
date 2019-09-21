import os
import yaml
import logging
import logging.config


def setup_logger(logger_file=None):
    with open(os.path.join(os.path.dirname(__file__),
                           'amlearn_logging.yaml'), 'r') as lf:
        config_dict = yaml.load(lf, Loader=yaml.FullLoader)
    if logger_file is not None:
        config_dict['handlers']['info_file_handler']['filename'] = logger_file
        for log_dict in config_dict['loggers'].values():
            log_dict['handlers'] += ['info_file_handler']

    logging.config.dictConfig(config_dict)


def get_logger(name):
    logging.basicConfig(format='"%(asctime)s | %(name)s | '
                               '%(levelname)s | %(message)s"')
    return logging.getLogger(name)
