import os
import yaml
import logging
import logging.config


def setup_logger(output_file=None):
    with open(os.path.join(os.path.dirname(__file__),
                           'amlearn_logging.yaml'), 'r') as lf:
        config_dict = yaml.load(lf)
    if output_file is not None:
        config_dict['handlers']['info_file_handler']['filename'] = output_file
    logging.config.dictConfig(config_dict)


def get_logger(name):
    logging.basicConfig(format='"%(asctime)s | %(name)s | '
                               '%(levelname)s | %(message)s"')
    return logging.getLogger(name)
