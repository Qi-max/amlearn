import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.base import BaseFeaturize
from amlearn.utils.check import check_featurizer_X

try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats error!")


class CNVoro(BaseFeaturize):
    def __init__(self, atoms_df=None, dependency="voro", nn_kwargs=None,
                 tmp_save=True, context=None):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(self.__class__, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             dependency=dependency,
                                             nn_kwargs=nn_kwargs,
                                             atoms_df=atoms_df)

    def fit(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        columns = X.columns
        if self._dependency.__class__.__name__ == "VoroNN" and \
                        'n_neighbors_voro' in columns:
            return self
        elif self._dependency.__class__.__name__ == "DistanceNN" and \
                        'n_neighbors_dist' in columns:
            return self
        else:
            self.atoms_df = self._dependency.fit_transform(X)
            return self

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        cn_list = np.zeros(len(X))

        n_neighbor_col = 'n_neighbors_dist' \
            if self._dependency.__class__.__name__ == "DistanceNN" \
            else 'n_neighbors_voro'
        cn_list = \
            voronoi_stats.cn_voro(cn_list, X[n_neighbor_col].values(),
                                  n_atoms=len(X))
        cn_list_df = pd.DataFrame(cn_list,
                                  index=range(len(X)),
                                  columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=cn_list_df,
                                                      name='cn_voro')

        return cn_list_df

    def get_feature_names(self):
        feature_names = ['CN_Dist'] \
            if self.dependency_name == 'dist' \
               or self.dependency_name == 'distance' \
            else ['CN_Voro']
        return feature_names
