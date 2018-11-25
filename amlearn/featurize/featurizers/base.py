import six
from amlearn.featurize.featurizers.voro_and_distance import VoroNN, DistanceNN
from amlearn.utils.backend import BackendContext, FeatureBackend
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


def create_featurizer_backend():
    backend_context = BackendContext()
    featurizer_backend = FeatureBackend(backend_context)
    return featurizer_backend


class BaseFeaturize(six.with_metaclass(ABCMeta,
                                       BaseEstimator, TransformerMixin)):
    def __init__(self,  atoms_df=None, tmp_save=True, context=None,
                 dependency=None, nn_kwargs=None):
        self.atoms_df = atoms_df
        self.tmp_save = tmp_save
        self.context = context if context is not None \
            else create_featurizer_backend()
        nn_kwargs = nn_kwargs if nn_kwargs else dict()
        if dependency is None:
            self._dependency = None
        elif isinstance(dependency, type):
            self.dependency_name = dependency.__class__.__name__.lower()[:-2]
            self._dependency = dependency
        elif isinstance(dependency, str):
            self.dependency_name = dependency
            if dependency == "voro" or dependency == "voronoi" :
                self._dependency = VoroNN(**nn_kwargs)
            elif dependency == "dist" or dependency == "distance":
                self._dependency = DistanceNN(**nn_kwargs)
            else:
                raise ValueError('dependency {} if unknown, Possible values '
                                 'are {}'.format(dependency,
                                                 '[voro, voronoi, '
                                                 'dist, distance]'))
        else:
            raise ValueError('dependency {} if unknown, Possible values '
                             'are {} or voro/dist object.'.format(
                              dependency, '[voro, voronoi, dist, distance]'))

    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @property
    def dependency(self):
        return self._dependency