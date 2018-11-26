import os
import pandas as pd
from amlearn.featurize.featurizers.mro import MRO
from amlearn.utils.basetest import AmLearnTest


module_dir = os.path.dirname(os.path.abspath(__file__))


class TestSro(AmLearnTest):
    def setUp(self):
        pass

    def test_cn_voro_from_dump_voro(self):
        atoms_df = pd.read_csv(os.path.join(module_dir, 'data',
                                            'sro', 'featurizer_voro_index.csv'),
                               index_col=0)
        mro = MRO(atoms_df=atoms_df)
        result_df = mro.fit_transform(X=None)
        print(result_df.head(5))
        # self.assertEqual(result_df.columns, ['CN_Voro'])
        self.assertEqual(len(result_df), len(atoms_df))
        # self.assertEqual(result_df['CN_Voro'].iloc[0], 15)
        # self.assertEqual(result_df['CN_Voro'].iloc[1], 13)
        # self.assertEqual(result_df['CN_Voro'].iloc[2], 16)

