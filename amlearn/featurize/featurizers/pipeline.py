import os
import inspect
import pkgutil
from operator import itemgetter
from amlearn.featurize.base import BaseFeaturize
from amlearn.featurize.featurizers.nearest_neighbor import BaseNN
from amlearn.featurize.featurizers.short_range_order import BaseInterstice
from amlearn.utils.check import is_abstract
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

module_dir = os.path.dirname(os.path.abspath(__file__))


def all_featurizers():
    all_classes = []
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=[module_dir], prefix='amlearn.featurize.featurizers.',
            onerror=lambda x: None):
        if ".tests." in modname or ".src." in modname:
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)
    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseFeaturize) and
                      c[0] != 'BaseFeaturize') or
                  ((issubclass(c[1], BaseNN) and c[0] != 'BaseNN'))]

    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]
    return sorted(set(estimators), key=itemgetter(0))


class MultiFeaturizer(BaseFeaturize):
    def __init__(self, featurizers="all", save=True, backend=None):
        super(MultiFeaturizer, self).__init__(
            save=save, backend=backend)
        self.featurizers = featurizers

    def fit(self, X=None):
        featurizer_dict = dict()
        if self.featurizers == "all":
            for featurizer in all_featurizers():
                instance = featurizer[1]()
                if instance.category in featurizer_dict.keys():
                    featurizers = featurizer_dict[instance.category]
                    featurizers.append(instance)
                else:
                    featurizer_dict[instance.category] = [instance]
        else:
            for featurizer in self.featurizers:
                if featurizer.category in featurizer_dict.keys():
                    featurizers = featurizer_dict[featurizer.category]
                    featurizers.append(featurizer)
                else:
                    featurizer_dict[featurizer.category] = [featurizer]
        self.featurizer_dict = featurizer_dict
        return self

    def transform(self, X, dependent_df=None):
        pipeline_list = []
        category_list = self.featurizer_dict.keys()
        if 'nearest_neighbor' in category_list:
            nn_pipeline = list()
            nn_pipeline.append(
                ('nearest_neighbor',
                 FeatureUnion([(instance.__class__.__name__, instance)
                               for instance in
                               self.featurizer_dict['nearest_neighbor']])
                 ))
            nn_pipeline.append(('final', FinalEstimator()))
            nn_pipe = Pipeline(nn_pipeline)
            nn_pipe.fit(X, y=None)
            dependent_df = nn_pipe.named_steps['final'].data

        if 'sro' in category_list:
            pipeline_list.append(
                ('sro',
                 FeatureUnion([(instance.__class__.__name__, instance)
                               for instance in
                               self.featurizer_dict['sro']])))
        if 'mro' in category_list:
            pipeline_list.append(
                ('mro', self.featurizer_dict['mro'][0]))
        pipeline_list.append(('final', FinalEstimator()))

        all_pipe = Pipeline(pipeline_list)
        all_pipe.fit(X, mro__dependent_df=dependent_df)
        X = all_pipe.named_steps['final'].data
        return X

    def get_feature_names(self):
        pass


class FinalEstimator(BaseEstimator):

    def fit(self, X, y=None):
        self._data = X
        return self

    @property
    def data(self):
        return self._data
