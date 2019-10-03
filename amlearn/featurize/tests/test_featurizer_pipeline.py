import pandas as pd
from amlearn.featurize.medium_range_order import MRO
from amlearn.featurize.nearest_neighbor import VoroNN, DistanceNN
from amlearn.featurize.short_range_order import \
    VolumeAreaInterstice, ClusterPackingEfficiency, AtomicPackingEfficiency, \
    BaseInterstice
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.pipeline import all_featurizers, \
    FeaturizePipeline


class TestSro(AmLearnTest):
    @classmethod
    def setUpClass(cls):
        cls.sc_df = \
            pd.DataFrame([[2, -0.0804011, -0.701738, -0.183609],
                          [1, 2.57287, -1.26719, 0.394576],
                          [1, -0.472962, 0.304242, -2.6716],
                          [2, -2.84262, 0.193787, -0.798494],
                          [1, -1.74748, 2.4581, -2.60968],
                          [1, 0.746941, -0.572718, 2.48917],
                          [1, -1.53659, -1.77562, -2.61867],
                          [1, 1.18244, -2.66335, -1.3482],
                          [2, 2.53219, 0.369375, -2.05735],
                          [1, -0.842491, 1.54805, 1.4869],
                          [1, -1.83099, -0.671698, 1.76418],
                          [1, -0.0331254, 2.00702, -0.819228],
                          [1, 1.67348, 1.15745, 0.789788]],
                         columns=['type', 'x', 'y', 'z'],
                         index=range(1, 14))
        cls.sc_bds = [[cls.sc_df['x'].min(), cls.sc_df['x'].max()],
                      [cls.sc_df['y'].min(), cls.sc_df['y'].max()],
                      [cls.sc_df['z'].min(), cls.sc_df['z'].max()]]

    def test_all_featurizers(self):
        featurizers = dict(all_featurizers())
        self.assertTrue('MRO' in featurizers.keys())
        self.assertTrue(issubclass(featurizers['VolumeAreaInterstice'],
                                   BaseInterstice))

    def test_featurize_pipeline(self):
        featurizers = [
            VoroNN(bds=self.sc_bds, cutoff=5, allow_neighbor_limit=300,
                   n_neighbor_limit=80, pbc=[1, 1, 1], save=True),
            DistanceNN(bds=self.sc_bds, cutoff=4, allow_neighbor_limit=300,
                       n_neighbor_limit=80, pbc=[1, 1, 1], save=True),
            VolumeAreaInterstice(
                atomic_number_list=[29, 40], save=True,
                radii=None, radius_type="miracle_radius", verbose=1),
            ClusterPackingEfficiency(
                atomic_number_list=[29, 40], save=True,
                radii=None, radius_type="miracle_radius", verbose=1),
            AtomicPackingEfficiency(
                atomic_number_list=[29, 40], save=True,
                radii=None, radius_type="miracle_radius", verbose=1),
            MRO(output_file_prefix='pipeline_mro')
        ]
        multi_featurizer = FeaturizePipeline(featurizers=featurizers)
        result_df = multi_featurizer.fit_transform(X=self.sc_df,
                                                   bds=self.sc_bds,
                                                   lammps_df=self.sc_df)
        print(result_df)


    # def test_all_featurizes_pipeline(self):
    #     featurize_pipeline = FeaturizePipeline()
    #     result_df = featurize_pipeline.fit_transform(X=self.sc_df)
    #     print(result_df)
