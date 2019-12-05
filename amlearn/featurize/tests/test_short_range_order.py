import os
import numpy as np
import pandas as pd
from amlearn.featurize.short_range_order import DistanceInterstice, \
    VolumeAreaInterstice, ClusterPackingEfficiency, AtomicPackingEfficiency, \
    CharacterMotif, IFoldSymmetry, AreaWtIFoldSymmetry, VolWtIFoldSymmetry, \
    VoroAreaStats, VoroAreaStatsSeparate, VoroVolStats, VoroVolStatsSeparate, \
    DistStats, BOOP, CN, VoroIndex
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.short_range_order import PackingOfSite

module_dir = os.path.dirname(os.path.abspath(__file__))



class TestSRO(AmLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.sc_voro=pd.read_pickle(os.path.join(module_dir, 'data',
                                                'voro_nn.pickle.gz'))
        cls.sc_dist=pd.read_pickle(os.path.join(module_dir, 'data',
                                                'dist_nn.pickle.gz'))
        cls.sc_df = pd.DataFrame([[2, -0.0804011, -0.7017380, -0.1836090],
                                  [1, 2.5728700, -1.2671900, 0.3945760],
                                  [1, -0.4729620, 0.3042420, -2.6716000],
                                  [2, -2.8426200, 0.1937870, -0.7984940],
                                  [1, -1.7474800, 2.4581000, -2.6096800],
                                  [1, 0.7469410, -0.5727180, 2.4891700],
                                  [1, -1.5365900, -1.7756200, -2.6186700],
                                  [1, 1.1824400, -2.6633500, -1.3482000],
                                  [2, 2.5321900, 0.3693750, -2.0573500],
                                  [1, -0.8424910, 1.5480500, 1.4869000],
                                  [1, -1.8309900, -0.6716980, 1.7641800],
                                  [1, -0.0331254, 2.0070200, -0.8192280],
                                  [1, 1.6734800, 1.1574500, 0.7897880]],
                                 columns=['type', 'x', 'y', 'z'],
                                 index=range(1, 14))
        cls.sc_bds = [[cls.sc_df['x'].min(), cls.sc_df['x'].max()],
                      [cls.sc_df['y'].min(), cls.sc_df['y'].max()],
                      [cls.sc_df['z'].min(), cls.sc_df['z'].max()]]

        cls.bcc_points = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ])
        cls.bcc_df = pd.DataFrame([[1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0],
                                   [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
                                   [1, 1, 1, 0], [1, 1, 1, 1]],
                                  columns=['type', 'x', 'y', 'z'])
        cls.bcc_radii = {str(x): {"atomic_radius": np.sqrt(3) / 4,
                                  "miracle_radius": np.sqrt(3) / 4}
                         for x in range(3, 6)}

        cls.fcc_points = np.array([
            [0, 0, 0], [-1, -1, 0], [0, 1, -1], [1, 1, 0], [0, 1, 1],
            [-1, 0, -1], [-1, 0, 1], [1, 0, 1], [1, 0, -1],
            [-1, -1, 0], [0, -1, 1], [1, -1, 0], [0, -1, -1]])
        cls.fcc_radii = {str(x): {"atomic_radius": np.sqrt(2)/2,
                                  "miracle_radius": np.sqrt(2)/2}
                         for x in range(3, 6)}

    def test_bcc_packing_of_site(self):
        neighbors_type = list()
        neighbors_coords = list()
        for x in self.bcc_points:
            neighbors_type.append('4')
            neighbors_coords.append(x)
        pos_ = PackingOfSite([1, 1, 1], [[0, 1], [0, 1], [0, 1]], '4',
                             np.array([.5, .5, .5]), neighbors_type,
                             neighbors_coords, radii=self.bcc_radii,
                             radius_type='miracle_radius')
        pos_.analyze_area_interstice()
        pos_.analyze_vol_interstice()
        self.assertEqual(pos_.volume_interstice_list_[0], 0.31982523841216826)
        self.assertEqual(pos_.area_interstice_list_[0], 0.4109513774519138)
        self.assertEqual(pos_.cluster_packing_efficiency(), 0.6801747615878317)
        self.assertEqual(pos_.atomic_packing_efficiency(), 0.383483)

    def test_fcc_packing_of_site(self):
        neighbors_type = list()
        neighbors_coords = list()
        for x in self.fcc_points[1:]:
            neighbors_type.append('4')
            neighbors_coords.append(x)
        pos_ = PackingOfSite([1, 1, 1], [[-1, 1], [-1, 1], [-1, 1]],
                             '4', self.fcc_points[0],
                             neighbors_type, neighbors_coords,
                             radii=self.fcc_radii, radius_type='miracle_radius')
        pos_.analyze_vol_interstice()
        pos_.analyze_area_interstice()
        self.assertSetEqual(set(pos_.volume_interstice_list_),
                            {0.27909705048253486, 0.2203644299557469,
                             0.24973074021914088})
        self.assertEqual(set(pos_.area_interstice_list_),
                         {0.2146018366025515, 0.09310031788289075,
                          0.44463963273020424, 0.2146018366025514,
                          0.4446396327302041})
        self.assertEqual(pos_.cluster_packing_efficiency(), 0.7437434130556606)
        self.assertEqual(pos_.atomic_packing_efficiency(), 0.09788699999999995)

    def test_distance_interstice(self):
        distance_interstice = DistanceInterstice(
            atomic_number_list=[29, 40], save=False,
            radii=None, radius_type="miracle_radius", verbose=1)

        df = distance_interstice.fit_transform(
            X=self.sc_voro, lammps_df = self.sc_df)
        self.assertEqual(len(df.columns), 5)
        self.assertAlmostEqual(df.iloc[2]['Dist_interstice_mean_voro'],
                               -0.078468, 6)
        self.assertAlmostEqual(df.iloc[2]['Dist_interstice_std_voro'],
                               0.185778, 6)
        self.assertAlmostEqual(df.iloc[2]['Dist_interstice_min_voro'],
                               -0.403807, 6)
        self.assertAlmostEqual(df.iloc[2]['Dist_interstice_max_voro'],
                               0.199099, 6)

    def test_volume_area_interstice(self):
        volume_area_interstice = VolumeAreaInterstice(
            pbc=[1,1,1], atomic_number_list=[29, 40], save=False,
            radii=None, radius_type="miracle_radius", verbose=1)

        df = volume_area_interstice.fit_transform(
            X=self.sc_voro, bds=self.sc_bds, lammps_df=self.sc_df)
        self.assertAlmostEqual(df.iloc[3]['Volume_interstice_mean_voro'],
                               0.197449, 6)
        self.assertAlmostEqual(df.iloc[3]['Volume_interstice_std_voro'],
                               0.172989, 6)
        self.assertAlmostEqual(df.iloc[3]['Volume_interstice_min_voro'],
                               0.000000, 6)
        self.assertAlmostEqual(df.iloc[3]['Volume_interstice_max_voro'],
                               0.497742, 6)
        self.assertAlmostEqual(df.iloc[3]['Area_interstice_mean_voro'],
                               0.278797, 6)
        self.assertAlmostEqual(df.iloc[3]['Area_interstice_std_voro'],
                               0.205535, 6)
        self.assertAlmostEqual(df.iloc[3]['Area_interstice_min_voro'],
                               0.000000, 6)
        self.assertAlmostEqual(df.iloc[3]['Area_interstice_max_voro'],
                               0.570205, 6)

    def test_cluster_packing_efficiency(self):
        cluster_packing_efficiency = ClusterPackingEfficiency(
            pbc=[1,1,1], atomic_number_list=[29, 40], save=False,
            radii=None, radius_type="miracle_radius", verbose=1)

        df = cluster_packing_efficiency.fit_transform(
            X=self.sc_voro, bds=self.sc_bds, lammps_df=self.sc_df)
        self.assertEqual(df.iloc[1, 0], 0.826991765889196)

    def test_atomic_pcking_efficiency(self):
        atomic_pcking_efficiency = AtomicPackingEfficiency(
            pbc=[1,1,1], atomic_number_list=[29, 40], save=False,
            radii=None, radius_type="miracle_radius", verbose=1)

        df = atomic_pcking_efficiency.fit_transform(
            X=self.sc_voro, bds=self.sc_bds, lammps_df=self.sc_df)
        self.assertEqual(df.iloc[2, 0], 0.05121967206477729)

    def test_cn_voro_from_voro(self):
        nn = CN(dependent_class="voro", save=False, backend=None)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('CN_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertEqual(df['CN_voro'].iloc[0], 10)
        self.assertEqual(df['CN_voro'].iloc[1], 11)
        self.assertEqual(df['CN_voro'].iloc[2], 11)

    def test_cn_voro_from_dist(self):
        nn = CN(dependent_class="dist", save=False, backend=None)
        df = nn.fit_transform(X=self.sc_dist)
        self.assertTrue('CN_dist' in df.columns)
        self.assertEqual(len(df), len(self.sc_dist))
        self.assertEqual(df['CN_dist'].iloc[0], 12)
        self.assertEqual(df['CN_dist'].iloc[1], 12)
        self.assertEqual(df['CN_dist'].iloc[2], 12)

    def test_voro_index(self):
        nn = VoroIndex(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Voronoi_idx_5_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertEqual(df['Voronoi_idx_4_voro'].iloc[0], 2)
        self.assertEqual(df['Voronoi_idx_5_voro'].iloc[0], 5)
        self.assertEqual(df['Voronoi_idx_5_voro'].iloc[2], 2)

    def test_character_motif(self):
        nn = CharacterMotif(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('is_polytetrahedral_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertEqual(df['is_polytetrahedral_voro'].iloc[0], 0)
        self.assertEqual(df['is_<0,0,12,0,0>_voro'].iloc[0], 0)
        self.assertEqual(df['is_polytetrahedral_voro'].iloc[9], 1)

    def test_i_fold_symmetry(self):
        nn = IFoldSymmetry(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('5_fold_symm_idx_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['7_fold_symm_idx_voro'].iloc[7],
                               0.1818, 4)
        self.assertAlmostEqual(df['5_fold_symm_idx_voro'].iloc[0],
                               0.5)
        self.assertAlmostEqual(df['5_fold_symm_idx_voro'].iloc[1],
                               0.2727, 4)

    def test_area_wt_i_fold_symmetry(self):
        nn = AreaWtIFoldSymmetry(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Area_wt_5_fold_symm_idx_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(
            df['Area_wt_5_fold_symm_idx_voro'].iloc[0], 0.4219, 4)
        self.assertAlmostEqual(
            df['Area_wt_6_fold_symm_idx_voro'].iloc[0], 0.3557, 4)
        self.assertAlmostEqual(
            df['Area_wt_4_fold_symm_idx_voro'].iloc[2], 0.1665, 4)

    def test_vol_wt_i_fold_symmetry(self):
        nn = VolWtIFoldSymmetry(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Vol_wt_7_fold_symm_idx_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(
            df['Vol_wt_5_fold_symm_idx_voro'].iloc[0], 0.4151, 4)
        self.assertAlmostEqual(
            df['Vol_wt_7_fold_symm_idx_voro'].iloc[0], 0.0000, 4)
        self.assertAlmostEqual(
            df['Vol_wt_7_fold_symm_idx_voro'].iloc[2], 0.1415, 4)

    def test_voronoi_area_stats(self):
        nn = VoroAreaStats(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Facet_area_sum_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['Facet_area_mean_voro'].iloc[0],
                               4.4345, 4)
        self.assertAlmostEqual(df['Facet_area_max_voro'].iloc[0],
                               10.2436, 4)
        self.assertAlmostEqual(df['Facet_area_std_voro'].iloc[2],
                               1.7594, 4)

    def test_voro_area_stats_seperate(self):
        nn = VoroAreaStatsSeparate(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('5_edged_area_mean_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['5_edged_area_mean_voro'].iloc[0],
                               3.7419, 4)
        self.assertAlmostEqual(df['4_edged_area_max_voro'].iloc[1],
                               1.3297, 4)
        self.assertAlmostEqual(df['5_edged_area_mean_voro'].iloc[2],
                               2.3947, 4)

    def test_voronoi_vol_stats(self):
        nn = VoroVolStats(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Subpolyhedra_vol_sum_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['Subpolyhedra_vol_std_voro'].iloc[0],
                               1.0287, 4)
        self.assertAlmostEqual(df['Subpolyhedra_vol_std_voro'].iloc[1],
                               1.3426, 4)
        self.assertAlmostEqual(df['Subpolyhedra_vol_max_voro'].iloc[2],
                               1.5079, 4)

    def test_voro_vol_stats_seperate(self):
        nn = VoroVolStatsSeparate(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('5_edged_vol_mean_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['5_edged_vol_sum_voro'].iloc[0],
                               8.3259, 4)
        self.assertAlmostEqual(df['5_edged_vol_sum_voro'].iloc[1],
                               4.8705, 4)
        self.assertAlmostEqual(df['4_edged_vol_std_voro'].iloc[2],
                               0.2231, 4)

    def test_dist_stats(self):
        nn = DistStats(save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('distance_std_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['distance_std_voro'].iloc[0],
                               0.1537, 4)
        self.assertAlmostEqual(df['distance_std_voro'].iloc[1],
                               0.4675, 4)
        self.assertAlmostEqual(df['distance_max_voro'].iloc[2],
                               3.0226, 4)

    def test_boop(self):
        atom_coords = self.sc_df[['x', 'y', 'z']].values.astype(np.longdouble)
        nn = BOOP(atom_coords=atom_coords, bds=self.sc_bds, save=False)
        df = nn.fit_transform(X=self.sc_voro)
        self.assertTrue('Coarse_grained_w_4_voro' in df.columns)
        self.assertEqual(len(df), len(self.sc_voro))
        self.assertAlmostEqual(df['q_4_voro'].iloc[0],
                               0.1990, 4)
        self.assertAlmostEqual(df['q_6_voro'].iloc[1],
                               0.3339, 4)
        self.assertAlmostEqual(df['q_8_voro'].iloc[2],
                               0.2384, 4)
