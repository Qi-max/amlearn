import os
import pandas as pd
from amlearn.featurize.featurizers.sro import CN, VoroIndex
from amlearn.utils.basetest import AmLearnTest


module_dir = os.path.dirname(os.path.abspath(__file__))


class TestSro(AmLearnTest):
    def setUp(self):
        pass
    #
    # def test_cn_voro_from_dump_voro(self):
    #     nn = CN.from_file(
    #         data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
    #         cutoff=4.2, allow_neighbor_limit=300,
    #         n_neighbor_limit=80, pbc=[1, 1, 1])
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN voro' in result_df.columns)
    #     self.assertEqual(len(result_df), 32000)
    #     self.assertEqual(result_df['CN voro'].iloc[0], 15)
    #     self.assertEqual(result_df['CN voro'].iloc[1], 13)
    #     self.assertEqual(result_df['CN voro'].iloc[2], 16)
    #
    # def test_cn_voro_from_dump_dist(self):
    #     nn = CN.from_file(
    #         data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
    #         cutoff=4.2, allow_neighbor_limit=300,
    #         n_neighbor_limit=80, pbc=[1, 1, 1],
    #         dependency="dist")
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN dist' in result_df.columns)
    #     self.assertEqual(len(result_df), 32000)
    #     self.assertEqual(result_df['CN dist'].iloc[0], 22)
    #     self.assertEqual(result_df['CN dist'].iloc[1], 22)
    #     self.assertEqual(result_df['CN dist'].iloc[2], 26)
    #
    # def test_cn_voro_from_voro(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = CN(atoms_df=atoms_df, dependency="voro",
    #             tmp_save=True, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['CN voro'].iloc[0], 15)
    #     self.assertEqual(result_df['CN voro'].iloc[1], 13)
    #     self.assertEqual(result_df['CN voro'].iloc[2], 16)

    def test_cn_voro_from_dist(self):
        atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
                                            'voro_and_distance',
                                            'featurizer_dist_nn.csv'),
                               index_col=0)
        nn = CN(atoms_df=atoms_df, dependency="dist",
                tmp_save=False, context=None)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('CN dist' in result_df.columns)
        self.assertEqual(len(result_df), len(atoms_df))
        self.assertEqual(result_df['CN dist'].iloc[0], 22)
        self.assertEqual(result_df['CN dist'].iloc[1], 22)
        self.assertEqual(result_df['CN dist'].iloc[2], 26)
    #
    # def test_voro_index(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = VoroIndex(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Voronoi idx5 voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['Voronoi idx4 voro'].iloc[0], 4)
    #     self.assertEqual(result_df['Voronoi idx5 voro'].iloc[0], 3)
    #     self.assertEqual(result_df['Voronoi idx5 voro'].iloc[2], 5)
