import numpy as np
from numpy import ma
import unittest

from .. import miscutils


class FindBlockTest(unittest.TestCase):
    def setUp(self):
        self.vector = [0, 1, 1, 1, 2, 3, 3, 4, 5, 5]
        self.array = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]]
        self.array_inds = [(0, 2), (3, 5)]
        self.array_vals = [np.array(self.array[0]), np.array(self.array[3])]

        self.masked_array = ma.masked_where(np.array(self.array) == 2, self.array)

        masked_arr2 = self.masked_array.copy()
        masked_arr2[1, 1] = 20
        masked_arr2[1, 1] = ma.masked
        self.masked_array2 = masked_arr2
        self.masked_array2_inds = [(3, 5)]

    def test_vector(self):
        inds, vals = miscutils.find_block(self.vector)
        self.assertEqual(inds, [(1, 4), (5, 7), (8, 10)])
        self.assertEqual(vals, [1, 3, 5])

    def test_vector_value(self):
        inds, vals = miscutils.find_block(self.vector, block_value=1)
        self.assertEqual(inds, [(1, 4)])
        self.assertEqual(vals, [1])

    def test_unmasked_array(self):
        inds, _ = miscutils.find_block(self.array)
        self.assertEqual(inds, self.array_inds)

    def test_unmasked_array_axis(self):
        a = np.array(self.array)
        inds, _ = miscutils.find_block(a, axis=1)
        self.assertEqual(inds, [])
        inds, _ = miscutils.find_block(a.transpose(), axis=1)
        self.assertEqual(inds, self.array_inds)

    def test_masked_array_no_ignore(self):
        inds, _ = miscutils.find_block(self.masked_array, ignore_masked=False)
        self.assertEqual(inds, self.array_inds)
        inds, _ = miscutils.find_block(self.masked_array2, ignore_masked=False)
        self.assertEqual(inds, self.masked_array2_inds)

    def test_masked_array_ignore(self):
        inds, _ = miscutils.find_block(self.masked_array, ignore_masked=True)
        self.assertEqual(inds, self.array_inds)
        inds, _ = miscutils.find_block(self.masked_array, ignore_masked=True)
        self.assertEqual(inds, self.array_inds)
