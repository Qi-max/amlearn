import os
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.sro import CN, VoroIndex, CharacterMotif, \
    IFoldSymmetry, AreaWtIFoldSymmetry, VolWtIFoldSymmetry, VoroAreaStats, \
    Boop, VoroAreaStatsSeparate
from amlearn.utils.basetest import AmLearnTest
from amlearn.utils.data import read_imd

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
    #
    # def test_cn_voro_from_dist(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_dist_nn.csv'),
    #                            index_col=0)
    #     nn = CN(atoms_df=atoms_df, dependency="dist",
    #             tmp_save=False, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN dist' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['CN dist'].iloc[0], 22)
    #     self.assertEqual(result_df['CN dist'].iloc[1], 22)
    #     self.assertEqual(result_df['CN dist'].iloc[2], 26)
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
    #
    # def test_character_motif(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = CharacterMotif(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('is polytetrahedral voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['is polytetrahedral voro'].iloc[0], 0)
    #     self.assertEqual(result_df['is <0,0,12,0,0> voro'].iloc[0], 0)
    #     self.assertEqual(result_df['is polytetrahedral voro'].iloc[9], 1)
    #
    # def test_i_fold_symmetry(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = IFoldSymmetry(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('5-fold symm idx' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['8-fold symm idx'].iloc[2], 0.0625)
    #     self.assertAlmostEqual(result_df['5-fold symm idx'].iloc[1], 0.2307692)
    #     self.assertAlmostEqual(result_df['5-fold symm idx'].iloc[2], 0.3125)
    #
    # def test_area_wt_i_fold_symmetry(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = AreaWtIFoldSymmetry(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Area_wt 8-fold symm idx' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertAlmostEqual(result_df['Area_wt 5-fold symm idx'].iloc[0],
    #                            0.1314728)
    #     self.assertAlmostEqual(result_df['Area_wt 6-fold symm idx'].iloc[0],
    #                            0.5387449)
    #     self.assertAlmostEqual(result_df['Area_wt 4-fold symm idx'].iloc[2],
    #                            0.0804660)
    #
    # def test_vol_wt_i_fold_symmetry(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = VolWtIFoldSymmetry(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Vol_wt 8-fold symm idx' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertAlmostEqual(result_df['Vol_wt 5-fold symm idx'].iloc[0],
    #                            0.1423263)
    #     self.assertAlmostEqual(result_df['Vol_wt 6-fold symm idx'].iloc[0],
    #                            0.5157505)
    #     self.assertAlmostEqual(result_df['Vol_wt 7-fold symm idx'].iloc[1],
    #                            0.0938408)
    #
    # def test_voronoi_area_stats(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     nn = VoroAreaStats(atoms_df=atoms_df)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Voronoi area' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertAlmostEqual(result_df['Facet area mean'].iloc[0],
    #                            1.7881286)
    #     self.assertAlmostEqual(result_df['Facet area max'].iloc[1],
    #                            2.9568438)
    #     self.assertAlmostEqual(result_df['Facet area std'].iloc[2],
    #                            0.9460697)

    def test_voro_area_stats_seperate(self):
        atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
                                            'voro_and_distance',
                                            'featurizer_voro_nn.csv'),
                               index_col=0)
        nn = VoroAreaStatsSeparate(atoms_df=atoms_df)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('5-edged area mean voro' in result_df.columns)
        self.assertEqual(len(result_df), len(atoms_df))
        self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[0],
                               1.1754510)
        self.assertAlmostEqual(result_df['4-edged area min voro'].iloc[1],
                               1.7194581)
        self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[2],
                               1.6061382)
    #
    # def test_boop(self):
    #     atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
    #                                         'voro_and_distance',
    #                                         'featurizer_voro_nn.csv'),
    #                            index_col=0)
    #     _, _, atom_coords, Bds = read_imd(os.path.join(module_dir,
    #                                                    'data',
    #                                                    'dump.overall.0'))
    #     atom_coords = np.array(atom_coords, dtype=np.float128)[:100, :]
    #     print(len(atom_coords))
    #     nn = Boop(atoms_df=atoms_df, atom_coords=atom_coords, Bds=Bds)
    #     result_df = nn.fit_transform(X=None)
    #     # self.assertTrue('Voronoi area' in result_df.columns)
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     # self.assertAlmostEqual(result_df['Facet area mean'].iloc[0],
    #     #                        1.7881286)
    #     # self.assertAlmostEqual(result_df['Facet area max'].iloc[1],
    #     #                        2.9568438)
    #     # self.assertAlmostEqual(result_df['Facet area std'].iloc[2],
    #     #                        0.9460697)

