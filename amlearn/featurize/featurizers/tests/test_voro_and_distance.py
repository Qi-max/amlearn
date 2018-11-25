import os
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.featurizers.voro_and_distance import VoroNN

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestVoro(AmLearnTest):
    def setUp(self):
        pass

    def test_voro(self):
        nn = VoroNN.from_file(
            data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
            cutoff=4.2, allow_neighbor_limit=300,
            n_neighbor_limit=80, pbc=[1, 1, 1])
        result_df = nn.fit_transform()

