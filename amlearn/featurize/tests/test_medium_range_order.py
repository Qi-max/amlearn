import os
import pandas as pd
from amlearn.featurize.medium_range_order import MRO
from amlearn.utils.basetest import AmLearnTest

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestMro(AmLearnTest):
    @classmethod
    def setUpClass(cls):
        cls.nn_df = pd.read_pickle(os.path.join(
            module_dir, 'data', 'voro_nn.pickle.gz'))

        cls.sro_volume_area_df = pd.read_pickle(os.path.join(
            module_dir, 'data', 'sro', 'feature_interstice_sro_voro_miracle_'
                                       'volume_area.pickle.gz'))
        cls.cluster_packing_df = pd.read_pickle(os.path.join(
            module_dir, 'data', 'sro', 'feature_sro_voro_miracle_cpe.pickle.gz')
        )
        cls.atomic_packing_df = pd.read_pickle(os.path.join(
            module_dir, 'data', 'sro', 'feature_sro_voro_miracle_ape.pickle.gz')
        )

    def test_volume_area_interstice_mro(self):
        mro = MRO(save=False)
        result_df = mro.fit_transform(X=self.sro_volume_area_df,
                                      dependent_df=self.nn_df)
        self.assertEqual(len(result_df.columns), 60)
        self.assertEqual(result_df.iloc[3, -1], -0.026077608080648407)

    def test_cluster_packing_mro(self):
        mro = MRO(save=False)
        result_df = mro.fit_transform(X=self.cluster_packing_df,
                                      dependent_df=self.nn_df)
        self.assertEqual(len(result_df.columns), 6)
        self.assertEqual(result_df.iloc[7, 0], 11.287025655788824)

    def test_atomic_packing_mro(self):
        mro = MRO(save=False)
        result_df = mro.fit_transform(X=self.atomic_packing_df,
                                      dependent_df=self.nn_df)
        self.assertEqual(len(result_df.columns), 6)
        self.assertEqual(result_df.iloc[5, 0], 1.6896773481062306)

    def test_all_interstice_and_packing_mro(self):
        all_df = self.cluster_packing_df.join(self.atomic_packing_df)
        all_df = all_df.join(self.sro_volume_area_df)
        mro = MRO(save=False)
        result_df = mro.fit_transform(X=all_df, dependent_df=self.nn_df)
        self.assertEqual(len(result_df.columns), 72)
        self.assertEqual(result_df.iloc[3, -1], -0.026077608080648407)
