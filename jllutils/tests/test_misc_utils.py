import numpy as np
from numpy import ma
import os
from pathlib import Path
import shutil
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


class FileTests(unittest.TestCase):
    real_test_path = Path(__file__).parent / 'file_test_dir'
    link_test_path = Path(__file__).parent / 'file_link_dir'
    real_test_file = real_test_path / 'real_file.txt'
    alt_test_file  = real_test_path / 'second_file.txt'
    link_test_file = real_test_path / 'link_file.txt'
    linkdir_real_test_file = link_test_path / 'real_file.txt'
    linkdir_alt_test_file  = link_test_path / 'second_file.txt'
    linkdir_link_test_file = link_test_path / 'link_file.txt'

    @classmethod
    def setUpClass(cls) -> None:
        cls.real_test_path.mkdir()
        os.symlink(cls.real_test_path, cls.link_test_path)
        with open(cls.real_test_file, 'w') as f:
            f.write('Testing File classes')
        with open(cls.alt_test_file, 'w') as f:
            f.write('Second File class test file')
        os.symlink(cls.real_test_file, cls.link_test_file)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.real_test_path)
        os.remove(cls.link_test_path)

    # ------------------ #
    # Tests for RealFile #
    # ------------------ #
    def test_realfile_same_file_equal(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.real_test_file)
        self.assertEqual(f1, f2)

    def test_realfile_link_equal(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.link_test_file)
        self.assertEqual(f1, f2)

    def test_realfile_same_file_equal_from_linked_dir(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.linkdir_real_test_file)
        self.assertEqual(f1, f2)

    def test_realfile_link_equal_from_linked_dir(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.linkdir_link_test_file)
        self.assertEqual(f1, f2)

    def test_realfile_diff_file_unequal(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.alt_test_file)
        self.assertNotEqual(f1, f2)

    def test_realfile_diff_link_unequal(self):
        f1 = miscutils.RealFile(self.link_test_file)
        f2 = miscutils.RealFile(self.alt_test_file)
        self.assertNotEqual(f1, f2)

    def test_realfile_diff_file_unequal_from_linked_dir(self):
        f1 = miscutils.RealFile(self.real_test_file)
        f2 = miscutils.RealFile(self.linkdir_alt_test_file)
        self.assertNotEqual(f1, f2)

    def test_realfile_diff_link_unequal_from_linked_dir(self):
        f1 = miscutils.RealFile(self.link_test_file)
        f2 = miscutils.RealFile(self.alt_test_file)
        self.assertNotEqual(f1, f2)

    # ------------------ #
    # Tests for LinkFile #
    # ------------------ #
    def test_linkfile_same_file_equal(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.real_test_file)
        self.assertEqual(f1, f2)

    def test_linkfile_link_not_equal(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.link_test_file)
        self.assertNotEqual(f1, f2)

    def test_linkfile_diff_file_not_equal(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.alt_test_file)
        self.assertNotEqual(f1, f2)

    def test_linkfile_same_file_equal_from_linked_dir(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.linkdir_real_test_file)
        self.assertEqual(f1, f2)

    def test_linkfile_link_not_equal_from_linked_dir(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.linkdir_link_test_file)
        self.assertNotEqual(f1, f2)

    def test_linkfile_diff_file_not_equal_from_linked_dir(self):
        f1 = miscutils.LinkFile(self.real_test_file)
        f2 = miscutils.LinkFile(self.linkdir_alt_test_file)
        self.assertNotEqual(f1, f2)
