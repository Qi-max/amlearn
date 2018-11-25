#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import find_packages
import warnings

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    msg = ("Please install numpy (version 1.7.0 or greater) before installing "
           "Amlearn. Because Amlearn uses numpy's f2py to compile the fortran "
           "modules to speed up the calculation of featurizer."
           " like:"
           "   $ conda install numpy"
           "   $ pip install numpy")
    raise RuntimeError(msg)


def calculate_version():
    initpy = open('amlearn/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version


package_version = calculate_version()
module_dir = os.path.dirname(os.path.abspath(__file__))

name = 'amlearn'
version = package_version
author = 'Qi Wang'
author_email = 'qwang.mse@gmail.com'
packages = find_packages()
package_data = {'amlearn.utils': ['*.yaml']}
url = 'https://github.com/Qi-max/amlearn'
download_url = 'https://github.com/Qi-max/amlearn/archive/0.0.1.tar.gz'
license = 'modified BSD'
description = 'Machine Learning package for amorphous materials.'
long_description = open(os.path.join(module_dir, 'README.md')).read()
zip_safe = False
install_requires = ['numpy>=1.7.0',
                    'scipy>=0.19.0',
                    'scikit-learn>=0.18.1',
                    'tqdm>=4.11.2',
                    'pandas>=0.20.2',
                    'six>=1.10.0']
ext_modules = [Extension(name='amlearn.fmodules',
                         sources=['amlearn/featurize/featurizers/sro_mro/utils.f90',
                                  'amlearn/featurize/featurizers/sro_mro/voronoi_nn.f90',
                                  'amlearn/featurize/featurizers/sro_mro/voronoi_stats.f90',
                                  'amlearn/featurize/featurizers/sro_mro/boop.f90',
                                  'amlearn/featurize/featurizers/sro_mro/mro_stats.f90',
                                  'amlearn/featurize/featurizers/sro_mro/neighbor_data.f90',
                                  'amlearn/featurize/featurizers/sro_mro/distance_stats.f90',
                                  'amlearn/featurize/featurizers/sro_mro/featurize.f90',
                                  'amlearn/featurize/featurizers/sro_mro/distance_nn.f90'])]
classifiers = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Fortran',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
]
keywords = ['amorphous materials', 'Materials Genome Initiative',
            'machine learning', 'data science', 'data mining',
            'AI', 'artificial intelligence',
            'featurizer', 'auto featurizer'
            'auto machine learning']
try:
    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        packages=packages,
        package_data=package_data,
        url=url,
        download_url=download_url,
        license=license,
        description=description,
        long_description=long_description,
        zip_safe=zip_safe,
        install_requires=install_requires,
        classifiers=classifiers,
        ext_modules=ext_modules,
        keywords=keywords
    )
except SystemExit as ex:
    if 'amlearn.fmodules' in ex.args[0]:
        warnings.warn('It looks like no fortran compiler is present. Retrying '
                      'installation without fortran modules.')
    else:
        raise ex
    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        packages=packages,
        package_data=package_data,
        url=url,
        download_url=download_url,
        license=license,
        description=description,
        long_description=long_description,
        zip_safe=zip_safe,
        install_requires=install_requires,
        classifiers=classifiers,
        ext_modules=[],
        keywords=keywords
    )
    warnings.warn('Installed amlearn without fortran modules since no fortran '
                  'compiler was found. The code may run slow as a result.')
