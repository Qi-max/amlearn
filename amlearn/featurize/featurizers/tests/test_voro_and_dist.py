import os
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.featurizers.voro_and_dist import VoroNN

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestVoro(AmLearnTest):
    def setUp(self):
        pass

    def test_voro(self):
        nn = VoroNN.from_file(
            data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
            cutoff=4.2, allow_neighbor_limit=300,
            n_neighbor_limit=80, pbc=[1, 1, 1], tmp_save=False)
        result_df = nn.fit_transform()
        self.assertEqual(len(result_df.columns), 401)
        self.assertEqual(len(result_df), 32000)
        self.assertEqual(result_df['n_neighbors_voro'].iloc[0], 15)
        self.assertEqual(result_df['neighbor_edge_79_voro'].iloc[1], 0)
        self.assertEqual(result_df['n_neighbors_voro'].iloc[1], 13)



