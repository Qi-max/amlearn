import os
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.sro import CN, VoroIndex, CharacterMotif, \
    IFoldSymmetry, AreaWtIFoldSymmetry, VolWtIFoldSymmetry, VoroAreaStats, \
    BOOP, VoroAreaStatsSeparate, VoroVolStats, VoroVolStatsSeparate, DistStats
from amlearn.utils.basetest import AmLearnTest
from amlearn.utils.data import read_imd

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestSRO(AmLearnTest):
    @classmethod
    def setUpClass(cls):
        cls.sc_voro = pd.read_csv(os.path.join(module_dir, 'data',
                                               'featurizer_voro_nn.csv'))
        cls.sc_dist = pd.read_csv(os.path.join(module_dir, 'data',
                                               'featurizer_dist_nn.csv'))
        cls.sc = pd.DataFrame([[2, -0.0804011, -0.701738, -0.183609],
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
                              columns=['type', 'x', 'y', 'z'])

        cls.sc_Bds = [[cls.sc['x'].min(), cls.sc['x'].max()],
                      [cls.sc['y'].min(), cls.sc['y'].max()],
                      [cls.sc['z'].min(), cls.sc['z'].max()]]

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
    #     nn = CN(atoms_df=self.sc_voro, dependency="voro",
    #             tmp_save=True, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertEqual(result_df['CN voro'].iloc[0], 10)
    #     self.assertEqual(result_df['CN voro'].iloc[1], 11)
    #     self.assertEqual(result_df['CN voro'].iloc[2], 11)

    def test_cn_voro_from_dist(self):
        nn = CN(atoms_df=self.sc_dist, dependency="dist",
                tmp_save=True, context=None)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('CN dist' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_dist))
        self.assertEqual(result_df['CN dist'].iloc[0], 12)
        self.assertEqual(result_df['CN dist'].iloc[1], 12)
        self.assertEqual(result_df['CN dist'].iloc[2], 12)

    def test_voro_index(self):
        nn = VoroIndex(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Voronoi idx5 voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertEqual(result_df['Voronoi idx4 voro'].iloc[0], 2)
        self.assertEqual(result_df['Voronoi idx5 voro'].iloc[0], 5)
        self.assertEqual(result_df['Voronoi idx5 voro'].iloc[2], 2)

    def test_character_motif(self):
        nn = CharacterMotif(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('is polytetrahedral voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertEqual(result_df['is polytetrahedral voro'].iloc[0], 0)
        self.assertEqual(result_df['is <0,0,12,0,0> voro'].iloc[0], 0)
        self.assertEqual(result_df['is polytetrahedral voro'].iloc[9], 1)

    def test_i_fold_symmetry(self):
        nn = IFoldSymmetry(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('5-fold symm idx voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['7-fold symm idx voro'].iloc[7],
                               0.1818, 4)
        self.assertAlmostEqual(result_df['5-fold symm idx voro'].iloc[0],
                               0.5)
        self.assertAlmostEqual(result_df['5-fold symm idx voro'].iloc[1],
                               0.2727, 4)

    def test_area_wt_i_fold_symmetry(self):
        nn = AreaWtIFoldSymmetry(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Area_wt 7-fold symm idx voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['Area_wt 5-fold symm idx voro'].iloc[0],
                               0.4219, 4)
        self.assertAlmostEqual(result_df['Area_wt 6-fold symm idx voro'].iloc[0],
                               0.3557, 4)
        self.assertAlmostEqual(result_df['Area_wt 4-fold symm idx voro'].iloc[2],
                               0.1665, 4)

    def test_vol_wt_i_fold_symmetry(self):
        nn = VolWtIFoldSymmetry(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Vol_wt 7-fold symm idx voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['Vol_wt 5-fold symm idx voro'].iloc[0],
                               0.4151, 4)
        self.assertAlmostEqual(result_df['Vol_wt 6-fold symm idx voro'].iloc[0],
                               0.3475, 4)
        self.assertAlmostEqual(result_df['Vol_wt 7-fold symm idx voro'].iloc[2],
                               0.1415, 4)

    def test_voronoi_area_stats(self):
        nn = VoroAreaStats(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Voronoi area voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['Facet area mean voro'].iloc[0],
                               4.4345, 4)
        self.assertAlmostEqual(result_df['Facet area max voro'].iloc[0],
                               10.2436, 4)
        self.assertAlmostEqual(result_df['Facet area std voro'].iloc[2],
                               1.7594, 4)

    def test_voro_area_stats_seperate(self):
        nn = VoroAreaStatsSeparate(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('5-edged area mean voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[0],
                               3.7419, 4)
        self.assertAlmostEqual(result_df['4-edged area max voro'].iloc[1],
                               1.3297, 4)
        self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[2],
                               2.3947, 4)

    def test_voronoi_vol_stats(self):
        nn = VoroVolStats(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Voronoi vol voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['Sub-polyhedra vol std voro'].iloc[0],
                               1.0287, 4)
        self.assertAlmostEqual(result_df['Sub-polyhedra vol std voro'].iloc[1],
                               1.3426, 4)
        self.assertAlmostEqual(result_df['Sub-polyhedra vol max voro'].iloc[2],
                               1.5079, 4)

    def test_voro_vol_stats_seperate(self):
        nn = VoroVolStatsSeparate(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('5-edged vol mean voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['5-edged vol sum voro'].iloc[0],
                               8.3259, 4)
        self.assertAlmostEqual(result_df['5-edged vol sum voro'].iloc[1],
                               4.8705, 4)
        self.assertAlmostEqual(result_df['4-edged vol std voro'].iloc[2],
                               0.2231, 4)

    def test_dist_stats(self):
        nn = DistStats(atoms_df=self.sc_voro)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Distance std voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['Distance std voro'].iloc[0],
                               0.1537, 4)
        self.assertAlmostEqual(result_df['Distance std voro'].iloc[1],
                               0.4675, 4)
        self.assertAlmostEqual(result_df['Distance max voro'].iloc[2],
                               3.0226, 4)

    def test_boop(self):
        atom_coords = self.sc[['x', 'y', 'z']].values.astype(np.float128)
        nn = BOOP(atoms_df=self.sc_voro, atom_coords=atom_coords,
                  Bds=self.sc_Bds)
        result_df = nn.fit_transform(X=None)
        self.assertTrue('Coarse-grained w_4 voro' in result_df.columns)
        self.assertEqual(len(result_df), len(self.sc_voro))
        self.assertAlmostEqual(result_df['q_4 voro'].iloc[0],
                               0.1990, 4)
        self.assertAlmostEqual(result_df['q_6 voro'].iloc[1],
                               0.3339, 4)
        self.assertAlmostEqual(result_df['q_8 voro'].iloc[2],
                               0.2384, 4)
