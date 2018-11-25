import os
import pandas as pd
from amlearn.featurize.featurizers.sro import CNVoro
from amlearn.utils.basetest import AmLearnTest
from amlearn.featurize.featurizers.voro_and_distance import VoroNN


module_dir = os.path.dirname(os.path.abspath(__file__))


class TestSro(AmLearnTest):
    def setUp(self):
        pass

    # def test_cn_voro_from_dump_voro(self):
    #     nn = CNVoro.from_file(
    #         data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
    #         cutoff=4.2, allow_neighbor_limit=300,
    #         n_neighbor_limit=80, pbc=[1, 1, 1])
    #     result_df = nn.fit_transform(X=None)
    #     self.assertEqual(result_df.columns, ['CN_Voro'])
    #     self.assertEqual(len(result_df), 32000)
    #     self.assertEqual(result_df['CN_Voro'].iloc[0], 15)
    #     self.assertEqual(result_df['CN_Voro'].iloc[1], 13)
    #     self.assertEqual(result_df['CN_Voro'].iloc[2], 16)

    def test_cn_voro_from_dump_dist(self):
        nn = CNVoro.from_file(
            data_path_file=os.path.join(module_dir, 'data', 'dump.overall.0'),
            cutoff=4.2, allow_neighbor_limit=300,
            n_neighbor_limit=80, pbc=[1, 1, 1],
            dependency="dist")
        result_df = nn.fit_transform(X=None)
        self.assertEqual(result_df.columns, ['CN_Dist'])
        self.assertEqual(len(result_df), 32000)
    #
    # def test_cn_voro_from_voro(self):
    #     atoms_df = pd.read_csv("/private/tmp/amlearn/task_13757/tmp_1543177006/featurizer/featurizer_voro_nn.csv",
    #                            index_col=0)
    #     nn = CNVoro(atoms_df=atoms_df, dependency="voro",
    #                 tmp_save=False, context=None)
    #     result_df = nn.fit_transform(X=None)
    #     self.assertEqual(result_df.columns, ['CN_Voro'])
    #     self.assertEqual(len(result_df), len(atoms_df))
    #     self.assertEqual(result_df['CN_Voro'].iloc[0], 15)
    #     self.assertEqual(result_df['CN_Voro'].iloc[1], 13)
    #     self.assertEqual(result_df['CN_Voro'].iloc[2], 16)
    #
