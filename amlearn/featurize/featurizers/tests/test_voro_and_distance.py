from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.featurizers.voro_and_distance import Voro


class TestVoro(AmLearnTest):
    def setUp(self):
        pass

    def test_voro(self):
        nn = Voro.from_file(
            data_path_file='/Users/Qi/Downloads/0.txt',
            cutoff=4.2, allow_neighbor_limit=300,
            n_neighbor_limit=80, pbc=[1, 1, 1])
        nn.transform()

