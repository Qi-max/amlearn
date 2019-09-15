import os
import pandas as pd
from amlearn.featurize.featurizers.nearest_neighbor import DistanceNN, VoroNN
from amlearn.utils.basetest import AmLearnTest

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestNearestNeighbor(AmLearnTest):
    @classmethod
    def setUpClass(cls):
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

    def test_voro(self):
        nn = VoroNN(Bds=self.sc_Bds, cutoff=5, allow_neighbor_limit=300,
                    n_neighbor_limit=80, pbc=[1, 1, 1], save=True)
        result_df = nn.fit_transform(self.sc)
        self.assertEqual(len(result_df.columns), 401)
        self.assertEqual(len(result_df), 13)
        self.assertEqual(result_df['n_neighbors_voro'].iloc[0], 10.0)
        self.assertEqual(result_df['n_neighbors_voro'].iloc[1], 11.0)
        self.assertEqual(result_df['neighbor_id_0_voro'].iloc[1], 1.0)

    def test_dist(self):
        nn = DistanceNN(Bds=self.sc_Bds, cutoff=4, allow_neighbor_limit=300,
                        n_neighbor_limit=80, pbc=[1, 1, 1], save=True)
        result_df = nn.fit_transform(self.sc)
        self.assertEqual(len(result_df.columns), 161)
        self.assertEqual(len(result_df), 13)
        self.assertEqual(result_df['n_neighbors_dist'].iloc[0], 12.0)
        self.assertEqual(result_df['n_neighbors_dist'].iloc[1], 12.0)
        self.assertEqual(result_df['neighbor_id_0_dist'].iloc[0], 2.0)


