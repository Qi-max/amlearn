import os
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.short_range_order import CN, VoroIndex, \
    CharacterMotif, IFoldSymmetry, AreaWtIFoldSymmetry, VolWtIFoldSymmetry, \
    VoroAreaStats, BOOP, VoroAreaStatsSeparate, VoroVolStats, \
    VoroVolStatsSeparate, DistStats, DistanceInterstice, VolumeAreaInterstice, \
    ClusterPackingEfficiency, AtomicPackingEfficiency
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.featurizers.short_range_order import PackingOfSite

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestSRO(AmLearnTest):
    @classmethod
    def setUpClass(cls):
        cls.sc_voro = pd.read_csv(os.path.join(module_dir, 'data',
                                               'featurizer_voro_nn.csv'))
        cls.sc_dist = pd.read_csv(os.path.join(module_dir, 'data',
                                               'featurizer_dist_nn.csv'))
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
        cls.sc_Bds = [[cls.sc_df['x'].min(), cls.sc_df['x'].max()],
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
        pos_.calculate_hull_facet_angles()
        pos_.calculate_hull_tetra_angles()
        pos_.analyze_hull_tetra_interstice()
        pos_.analyze_hull_facet_interstice()
        pos_.combine_neighbor_solid_angles()
        self.assertEqual(pos_.volume_interstice_list_[0], 0.31982523841216826)
        self.assertEqual(pos_.area_interstice_list_[0], 0.41095137745191357)
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
        pos_.calculate_hull_facet_angles()
        pos_.calculate_hull_tetra_angles()
        pos_.analyze_hull_tetra_interstice()
        pos_.analyze_hull_facet_interstice()
        pos_.combine_neighbor_solid_angles()
        print(pos_.cluster_packing_efficiency())
        print(pos_.atomic_packing_efficiency())
        print(set(pos_.volume_interstice_list_))
        print(set(pos_.area_interstice_list_))
        self.assertSetEqual(set(pos_.volume_interstice_list_),
                            {0.27909705048253486, 0.2203644299557469,
                             0.24973074021914088})
        self.assertEqual(set(pos_.area_interstice_list_),
                         {0.21460183660255128, 0.09310031788289119,
                          0.444639632730204, 0.21460183660255117,
                          0.4446396327302041})
        self.assertEqual(pos_.cluster_packing_efficiency(), 0.7437434130556606)
        self.assertEqual(pos_.atomic_packing_efficiency(), 0.09788699999999995)

    def test_distance_interstice(self):
        distance_interstice = DistanceInterstice(
            type_to_atomic_number_list=[29, 40], save=True,
            radii=None, radius_type="miracle_radius", verbose=1)

        result_df = distance_interstice.fit_transform(
            X=pd.read_pickle(os.path.join(module_dir, 'data',
                                          'featurizer_voro_nn.pickle.gz')),
            lammps_df = self.sc_df)
        self.assertEqual(len(result_df.columns), 5)
        self.assertEqual(result_df.iloc[2, 0], 10.136853319891685)

    def test_volume_area_interstice(self):
        volume_area_interstice = VolumeAreaInterstice(
            pbc=[1,1,1], type_to_atomic_number_list=[29, 40], save=True,
            radii=None, radius_type="miracle_radius", verbose=1)

        result_df = volume_area_interstice.fit_transform(
            X=pd.read_pickle(os.path.join(module_dir, 'data',
                                          'featurizer_voro_nn.pickle.gz')),
            Bds=self.sc_Bds, lammps_df=self.sc_df)
        self.assertEqual(result_df.iloc[3, 0], 1.7205243759077298)
        self.assertEqual(result_df.iloc[1, -1], 0.6671266923318973)

    def test_cluster_packing_efficiency(self):
        cluster_packing_efficiency = ClusterPackingEfficiency(
            pbc=[1,1,1], type_to_atomic_number_list=[29, 40], save=True,
            radii=None, radius_type="miracle_radius", verbose=1)

        result_df = cluster_packing_efficiency.fit_transform(
            X=pd.read_pickle(os.path.join(module_dir, 'data',
                                          'featurizer_voro_nn.pickle.gz')),
            Bds=self.sc_Bds, lammps_df=self.sc_df)
        self.assertEqual(result_df.iloc[1, 0], 0.826991765889196)

    def test_atomic_pcking_efficiency(self):
        atomic_pcking_efficiency = AtomicPackingEfficiency(
            pbc=[1,1,1], type_to_atomic_number_list=[29, 40], save=True,
            radii=None, radius_type="miracle_radius", verbose=1)

        result_df = atomic_pcking_efficiency.fit_transform(
            X=pd.read_pickle(os.path.join(module_dir, 'data',
                                          'featurizer_voro_nn.pickle.gz')),
            Bds=self.sc_Bds, lammps_df=self.sc_df)
        self.assertEqual(result_df.iloc[2, 0], 0.05121967206477729)

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
    #     nn = CN(atoms_df=self.sc_voro, dependency="voro",
    #             tmp_save=True, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertEqual(result_df['CN voro'].iloc[0], 10)
    #     self.assertEqual(result_df['CN voro'].iloc[1], 11)
    #     self.assertEqual(result_df['CN voro'].iloc[2], 11)
    #
    # def test_cn_voro_from_dist(self):
    #     nn = CN(atoms_df=self.sc_dist, dependency="dist",
    #             tmp_save=True, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('CN dist' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_dist))
    #     self.assertEqual(result_df['CN dist'].iloc[0], 12)
    #     self.assertEqual(result_df['CN dist'].iloc[1], 12)
    #     self.assertEqual(result_df['CN dist'].iloc[2], 12)
    #
    # def test_voro_index(self):
    #     nn = VoroIndex(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Voronoi idx5 voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertEqual(result_df['Voronoi idx4 voro'].iloc[0], 2)
    #     self.assertEqual(result_df['Voronoi idx5 voro'].iloc[0], 5)
    #     self.assertEqual(result_df['Voronoi idx5 voro'].iloc[2], 2)
    #
    # def test_character_motif(self):
    #     nn = CharacterMotif(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('is polytetrahedral voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertEqual(result_df['is polytetrahedral voro'].iloc[0], 0)
    #     self.assertEqual(result_df['is <0,0,12,0,0> voro'].iloc[0], 0)
    #     self.assertEqual(result_df['is polytetrahedral voro'].iloc[9], 1)
    #
    # def test_i_fold_symmetry(self):
    #     nn = IFoldSymmetry(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('5-fold symm idx voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['7-fold symm idx voro'].iloc[7],
    #                            0.1818, 4)
    #     self.assertAlmostEqual(result_df['5-fold symm idx voro'].iloc[0],
    #                            0.5)
    #     self.assertAlmostEqual(result_df['5-fold symm idx voro'].iloc[1],
    #                            0.2727, 4)
    #
    # def test_area_wt_i_fold_symmetry(self):
    #     nn = AreaWtIFoldSymmetry(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Area_wt 7-fold symm idx voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['Area_wt 5-fold symm idx voro'].iloc[0],
    #                            0.4219, 4)
    #     self.assertAlmostEqual(result_df['Area_wt 6-fold symm idx voro'].iloc[0],
    #                            0.3557, 4)
    #     self.assertAlmostEqual(result_df['Area_wt 4-fold symm idx voro'].iloc[2],
    #                            0.1665, 4)
    #
    # def test_vol_wt_i_fold_symmetry(self):
    #     nn = VolWtIFoldSymmetry(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Vol_wt 7-fold symm idx voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['Vol_wt 5-fold symm idx voro'].iloc[0],
    #                            0.4151, 4)
    #     self.assertAlmostEqual(result_df['Vol_wt 6-fold symm idx voro'].iloc[0],
    #                            0.3475, 4)
    #     self.assertAlmostEqual(result_df['Vol_wt 7-fold symm idx voro'].iloc[2],
    #                            0.1415, 4)
    #
    # def test_voronoi_area_stats(self):
    #     nn = VoroAreaStats(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Voronoi area voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['Facet area mean voro'].iloc[0],
    #                            4.4345, 4)
    #     self.assertAlmostEqual(result_df['Facet area max voro'].iloc[0],
    #                            10.2436, 4)
    #     self.assertAlmostEqual(result_df['Facet area std voro'].iloc[2],
    #                            1.7594, 4)
    #
    # def test_voro_area_stats_seperate(self):
    #     nn = VoroAreaStatsSeparate(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('5-edged area mean voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[0],
    #                            3.7419, 4)
    #     self.assertAlmostEqual(result_df['4-edged area max voro'].iloc[1],
    #                            1.3297, 4)
    #     self.assertAlmostEqual(result_df['5-edged area mean voro'].iloc[2],
    #                            2.3947, 4)
    #
    # def test_voronoi_vol_stats(self):
    #     nn = VoroVolStats(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Voronoi vol voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['Sub-polyhedra vol std voro'].iloc[0],
    #                            1.0287, 4)
    #     self.assertAlmostEqual(result_df['Sub-polyhedra vol std voro'].iloc[1],
    #                            1.3426, 4)
    #     self.assertAlmostEqual(result_df['Sub-polyhedra vol max voro'].iloc[2],
    #                            1.5079, 4)
    #
    # def test_voro_vol_stats_seperate(self):
    #     nn = VoroVolStatsSeparate(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('5-edged vol mean voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['5-edged vol sum voro'].iloc[0],
    #                            8.3259, 4)
    #     self.assertAlmostEqual(result_df['5-edged vol sum voro'].iloc[1],
    #                            4.8705, 4)
    #     self.assertAlmostEqual(result_df['4-edged vol std voro'].iloc[2],
    #                            0.2231, 4)
    #
    # def test_dist_stats(self):
    #     nn = DistStats(atoms_df=self.sc_voro)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Distance std voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['Distance std voro'].iloc[0],
    #                            0.1537, 4)
    #     self.assertAlmostEqual(result_df['Distance std voro'].iloc[1],
    #                            0.4675, 4)
    #     self.assertAlmostEqual(result_df['Distance max voro'].iloc[2],
    #                            3.0226, 4)
    #
    # def test_boop(self):
    #     atom_coords = self.sc[['x', 'y', 'z']].values.astype(np.longdouble)
    #     nn = BOOP(atoms_df=self.sc_voro, atom_coords=atom_coords,
    #               Bds=self.sc_Bds)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertTrue('Coarse-grained w_4 voro' in result_df.columns)
    #     self.assertEqual(len(result_df), len(self.sc_voro))
    #     self.assertAlmostEqual(result_df['q_4 voro'].iloc[0],
    #                            0.1990, 4)
    #     self.assertAlmostEqual(result_df['q_6 voro'].iloc[1],
    #                            0.3339, 4)
    #     self.assertAlmostEqual(result_df['q_8 voro'].iloc[2],
    #                            0.2384, 4)
