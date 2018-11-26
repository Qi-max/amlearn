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
        self.assertTrue('Voronoi idx5 voro sum_NN' in result_df.columns)
        self.assertEqual(len(result_df), len(atoms_df))
        # self.assertAlmostEqual(result_df['Voronoi idx3 voro sum_NN'].iloc[0],
        #                        21445)
        # self.assertAlmostEqual(result_df['Voronoi idx4 voro std_NN'].iloc[12],
        #                        5522.932)
        # self.assertAlmostEqual(result_df['Voronoi idx4 voro diff_NN'].iloc[13],
        #                        -3)

