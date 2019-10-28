import os
import pandas as pd
import unittest
from amlearn.featurize.symmetry_function import BPRadialFunction, BPAngularFunction
from amlearn.utils.basetest import AmLearnTest


class TestSymmetryFunction(AmLearnTest):
    @classmethod
    def setUpClass(cls):
        cls.sc_binary = pd.DataFrame([[1, 2, -0.0804011, -0.701738, -0.183609],
                                      [2, 1, 2.57287, -1.26719, 0.394576],
                                      [3, 1, -0.472962, 0.304242, -2.6716],
                                      [4, 2, -2.84262, 0.193787, -0.798494],
                                      [5, 1, -1.74748, 2.4581, -2.60968],
                                      [6, 1, 0.746941, -0.572718, 2.48917],
                                      [7, 1, -1.53659, -1.77562, -2.61867],
                                      [8, 1, 1.18244, -2.66335, -1.3482],
                                      [9, 2, 2.53219, 0.369375, -2.05735],
                                      [10, 1, -0.842491, 1.54805, 1.4869],
                                      [11, 1, -1.83099, -0.671698, 1.76418],
                                      [12, 1, -0.0331254, 2.00702, -0.819228],
                                      [13, 1, 1.67348, 1.15745, 0.789788]],
                                     columns=['id', 'type', 'x', 'y', 'z'],
                                     index=range(1, 14))
        cls.sc_binary_bds = [[cls.sc_binary['x'].min(), cls.sc_binary['x'].max()],
                             [cls.sc_binary['y'].min(), cls.sc_binary['y'].max()],
                             [cls.sc_binary['z'].min(), cls.sc_binary['z'].max()]]

        cls.sc_ternary = pd.DataFrame([[1, 2, -0.0804011, -0.701738, -0.183609],
                                       [2, 1, 2.57287, -1.26719, 0.394576],
                                       [3, 1, -0.472962, 0.304242, -2.6716],
                                       [4, 2, -2.84262, 0.193787, -0.798494],
                                       [5, 1, -1.74748, 2.4581, -2.60968],
                                       [6, 3, 0.746941, -0.572718, 2.48917],
                                       [7, 1, -1.53659, -1.77562, -2.61867],
                                       [8, 1, 1.18244, -2.66335, -1.3482],
                                       [9, 2, 2.53219, 0.369375, -2.05735],
                                       [10, 1, -0.842491, 1.54805, 1.4869],
                                       [11, 3, -1.83099, -0.671698, 1.76418],
                                       [12, 1, -0.0331254, 2.00702, -0.819228],
                                       [13, 1, 1.67348, 1.15745, 0.789788]],
                                      columns=['id', 'type', 'x', 'y', 'z'],
                                      index=range(1, 14))
        cls.sc_ternary_bds = [[cls.sc_binary['x'].min(), cls.sc_binary['x'].max()],
                              [cls.sc_binary['y'].min(), cls.sc_binary['y'].max()],
                              [cls.sc_binary['z'].min(), cls.sc_binary['z'].max()]]

    def test_bp_radial_binary(self):
        bpr = BPRadialFunction(bds=self.sc_binary_bds, atom_type_symbols=[1, 2],
                               pbc=[1, 1, 1], delta_r=0.2, n_r=20, cutoff=6.5,
                               save=False)
        df = bpr.fit_transform(self.sc_binary)
        self.assertEqual(len(df.columns), 40)
        self.assertEqual(len(df), 13)
        self.assertAlmostEqual(df["1_1.200"].loc[1], 0, 5)
        self.assertAlmostEqual(df["1_2.000"].loc[1], 0.076, 3)
        self.assertAlmostEqual(df["1_2.600"].loc[1], 6.606, 3)
        self.assertAlmostEqual(df["1_2.800"].loc[1], 6.518, 3)
        self.assertAlmostEqual(df["1_3.800"].loc[1], 0.423, 3)

    def test_bp_radial_ternary(self):
        bpr = BPRadialFunction(bds=self.sc_ternary_bds,
                               atom_type_symbols=[1, 2, 3],
                               pbc=[1, 1, 1], delta_r=0.2, n_r=20, cutoff=6.5,
                               save=False)
        df = bpr.fit_transform(self.sc_ternary)
        self.assertEqual(len(df.columns), 60)
        self.assertEqual(len(df), 13)
        self.assertAlmostEqual(df["1_1.200"].loc[1], 0, 5)
        self.assertAlmostEqual(df["1_2.000"].loc[1], 0.060, 3)
        self.assertAlmostEqual(df["1_2.600"].loc[1], 4.618, 3)
        self.assertAlmostEqual(df["1_2.800"].loc[1], 5.171, 3)
        self.assertAlmostEqual(df["1_3.800"].loc[1], 0.423, 3)

        self.assertAlmostEqual(df["2_1.200"].loc[1], 0, 5)
        self.assertAlmostEqual(df["2_2.000"].loc[1], 0, 3)
        self.assertAlmostEqual(df["2_2.800"].loc[1], 0.959, 3)
        self.assertAlmostEqual(df["2_3.400"].loc[1], 1.027, 3)
        self.assertAlmostEqual(df["2_3.800"].loc[1], 0.121, 3)

        self.assertAlmostEqual(df["3_1.200"].loc[1], 0, 5)
        self.assertAlmostEqual(df["3_2.000"].loc[1], 0.016, 3)
        self.assertAlmostEqual(df["3_2.600"].loc[1], 1.988, 3)
        self.assertAlmostEqual(df["3_3.400"].loc[1], 0.001, 3)
        self.assertAlmostEqual(df["3_3.800"].loc[1], 0, 3)

    def test_bp_angular_binary(self):
        ksaais, lambdas, zetas = [2.6, 3.5], [-1, 1], [1, 1]
        bpa = BPAngularFunction(bds=self.sc_binary_bds, atom_type_symbols=[1, 2],
                                ksaais=ksaais, lambdas=lambdas, zetas=zetas,
                                pbc=[1, 1, 1], save=False)
        df = bpa.fit_transform(self.sc_binary)
        self.assertEqual(len(df.columns), 6)
        self.assertEqual(len(df), 13)
        self.assertAlmostEqual(df["1_1_2.600_-1.000_1.000"].loc[1], 1.681, 3)
        self.assertAlmostEqual(df["1_1_3.500_1.000_1.000"].loc[1], 6.992, 3)
        self.assertAlmostEqual(df["1_2_2.600_-1.000_1.000"].loc[1], 0.434, 3)
        self.assertAlmostEqual(df["1_2_3.500_1.000_1.000"].loc[1], 2.997, 3)
        self.assertAlmostEqual(df["2_2_2.600_-1.000_1.000"].loc[1], 0.003, 3)
        self.assertAlmostEqual(df["2_2_3.500_1.000_1.000"].loc[1], 0.339, 3)

    def test_bp_angular_ternary(self):
        ksaais, lambdas, zetas = [2.6, 3.5], [-1, 1], [1, 1]
        bpa = BPAngularFunction(bds=self.sc_ternary_bds,
                                atom_type_symbols=[1, 2, 3],
                                ksaais=ksaais, lambdas=lambdas, zetas=zetas,
                                pbc=[1, 1, 1], save=False)
        df = bpa.fit_transform(self.sc_ternary)
        self.assertEqual(len(df.columns), 12)
        self.assertEqual(len(df), 13)
        self.assertAlmostEqual(df["1_1_2.600_-1.000_1.000"].loc[1], 0.928, 3)
        self.assertAlmostEqual(df["1_1_3.500_1.000_1.000"].loc[1], 4.226, 3)
        self.assertAlmostEqual(df["1_2_2.600_-1.000_1.000"].loc[1], 0.283, 3)
        self.assertAlmostEqual(df["1_2_3.500_1.000_1.000"].loc[1], 2.435, 3)
        self.assertAlmostEqual(df["1_3_2.600_-1.000_1.000"].loc[1], 0.666, 3)
        self.assertAlmostEqual(df["1_3_3.500_1.000_1.000"].loc[1], 2.750, 3)
        self.assertAlmostEqual(df["2_2_2.600_-1.000_1.000"].loc[1], 0.003, 3)
        self.assertAlmostEqual(df["2_2_3.500_1.000_1.000"].loc[1], 0.339, 3)
        self.assertAlmostEqual(df["2_3_2.600_-1.000_1.000"].loc[1], 0.151, 3)
        self.assertAlmostEqual(df["2_3_3.500_1.000_1.000"].loc[1], 0.562, 3)
        self.assertAlmostEqual(df["3_3_2.600_-1.000_1.000"].loc[1], 0.087, 3)
        self.assertAlmostEqual(df["3_3_3.500_1.000_1.000"].loc[1], 0.015, 3)

if __name__ == '__main__':
    unittest.main()