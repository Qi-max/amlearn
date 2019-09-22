import os
import inspect
import pkgutil
import pandas as pd
from operator import itemgetter
from amlearn.featurize.base import BaseFeaturize
from amlearn.featurize.featurizers.nearest_neighbor import BaseNN
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


class FeaturizerPipeline(BaseFeaturize):
    def __init__(self, featurizers="all", save=True, backend=None,
                 output_path=None):
        super(FeaturizerPipeline, self).__init__(
            save=save, backend=backend, output_path=output_path)
        self.featurizers = featurizers

    def fit(self, X=None, dependent_df=None, lammps_df=None, bds=None,
            lammps_path=None):
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
        self.dependent_df = dependent_df
        self.lammps_df = lammps_df
        self.bds=bds
        self.lammps_path=lammps_path
        return self

    def transform(self, X):
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
            nn_pipe.fit(X)
            dependent_df = pd.DataFrame(
                nn_pipe.named_steps['final'].data, index=X.index,
                columns=[name.split("__")[1] for name in
                         nn_pipe.named_steps[
                             'nearest_neighbor'].get_feature_names()])
            nn_df = dependent_df
        else:
            dependent_df = self.dependent_df
            nn_df = X

        if 'sro' in category_list:
            sro_pipeline = list()
            sro_pipeline.append(
                ('sro',
                 FeatureUnion([('{}_{}'.format(instance.__class__.__name__,
                                               instance.dependent_class),
                                instance) for instance in
                               self.featurizer_dict['sro']])))
            sro_pipeline.append(('final', FinalEstimator()))
            sro_pipe = Pipeline(sro_pipeline)
            sro_pipe.fit(nn_df)
            common_sro_df = pd.DataFrame(
                sro_pipe.named_steps['final'].data, index=X.index,
                columns=[name.split("__")[1] for name in
                         sro_pipe.named_steps['sro'].get_feature_names()])
        else:
            common_sro_df = None

        if 'interstice_sro' in category_list:
            interstice_sro_pipeline = list()
            interstice_sro_pipeline.append(
                ('interstice_sro',
                 FeatureUnion([('{}_{}'.format(instance.__class__.__name__,
                                               instance.dependent_class),
                                instance) for instance in
                               self.featurizer_dict['interstice_sro']])))
            interstice_sro_pipeline.append(('final', FinalEstimator()))
            interstice_sro_pipe = Pipeline(interstice_sro_pipeline)
            interstice_sro_pipe.fit(
                nn_df, interstice_sro__lammps_df=self.lammps_df,
                interstice_sro__lammps_path=self.lammps_path,
                interstice_sro__bds=self.bds)
            interstice_sro_df = pd.DataFrame(
                interstice_sro_pipe.named_steps['final'].data, index=X.index,
                columns=[name.split("__")[1] for name in
                         interstice_sro_pipe.named_steps[
                             'interstice_sro'].get_feature_names()])
        else:
            interstice_sro_df = None

        sro_df = common_sro_df.join(interstice_sro_df) \
            if common_sro_df is not None and interstice_sro_df is not None \
            else common_sro_df if common_sro_df is not None \
            else interstice_sro_df if interstice_sro_df is not None else nn_df

        if 'mro' in category_list:
            mro_list = []
            mro_list.append(
                ('mro', self.featurizer_dict['mro'][0]))
            mro_list.append(('final', FinalEstimator()))
            mro_pipe = Pipeline(mro_list)
            mro_pipe.fit(sro_df, mro__dependent_df=dependent_df)
            mro_df = pd.DataFrame(
                mro_pipe.named_steps['final'].data, index=X.index,
                columns=mro_pipe.named_steps['mro'].get_feature_names())
        else:
            mro_df = None
        if sro_df is not X and mro_df is not None:
            self.backend.save_featurizer_as_dataframe(sro_df.join(mro_df),
                                                      name='features_all')
            return sro_df.join(mro_df)
        return mro_df if mro_df is not None else sro_df

    def get_feature_names(self):
        pass


class FinalEstimator(BaseEstimator):

    def fit(self, X, y=None):
        self._data = X
        return self

    @property
    def data(self):
        return self._data
