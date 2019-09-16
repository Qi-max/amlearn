import numpy as np
from amlearn.utils.basetest import AmLearnTest
from amlearn.utils.data import get_isometric_lists


class test_data(AmLearnTest):
    def setUp(self):
        pass

    def test_get_isometric_lists(self):
        test_lists= [[1, 2, 3], [4], [5, 6], [1, 2, 3]]
        isometric_lists = \
            get_isometric_lists(test_lists, limit_width=80, fill_value=0)
        self.assertEqual(np.array(isometric_lists).shape, (4, 80))

        test_arrays = np.array([np.array([1, 2, 3]), np.array([4]),
                               np.array([5, 6]), np.array([1, 2, 3])])
        isometric_arrays = \
            get_isometric_lists(test_arrays, limit_width=80, fill_value=0)
        self.assertEqual(np.array(isometric_arrays).shape, (4, 80))

