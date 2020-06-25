import os
import unittest
import shutil
from pathlib import Path

from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

class LoadSaveUtilsTestsTests(unittest.TestCase):
    """
    Tests: list_files_in_dir and list_dirs_in_dir
    """

    def setUp(self):
        """
        Setup
        """
        directory = os.path.dirname(__file__)
        self.directory = os.path.join(directory, 'load_save_utils_tests_data')
        shutil.rmtree(self.directory, ignore_errors=True)
        Path(self.directory).mkdir()

        self.files_txt = [os.path.join(self.directory, f"{i}.txt") for i in range(5)]
        for filename in self.files_txt:
            Path(filename).touch()
        
        self.files_pkl = [os.path.join(self.directory, f"{i}.pkl") for i in range(5)]
        for filename in self.files_pkl:
            Path(filename).touch()
        
        self.directories = [os.path.join(self.directory, f"{i}") for i in range(5)]
        self.directories.append(os.path.join(self.directory, '.git'))
        for filename in self.directories:
            Path(filename).mkdir()


    def tearDown(self):
        """
        Tear down
        """
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_list_valid_files(self):
        """
        Test: Files are correct listed
        """
        self.assertEqual(list_files_in_dir(self.directory, '.txt').sort(), self.files_txt.sort())
        self.assertEqual(list_files_in_dir(self.directory, '.pkl').sort(), self.files_pkl.sort())

        files_all = self.files_txt
        files_all.extend(self.files_pkl)
        self.assertEqual(list_files_in_dir(self.directory).sort(), files_all.sort())

        self.assertEqual(list_files_in_dir(self.directories[0]), [])

    def test_list_files_in_file(self):
        """
        Test: List a file instead of a directory
        """
        with self.assertRaises(AssertionError):
            list_files_in_dir(self.files_txt[0])
    

    def test_list_valid_directories(self):
        """
        Test: Directories are correct listed
        """
        dirs = list_dirs_in_dir(self.directory, include_git=True)
        for dir in dirs:
            assert dir in self.directories
        for dir in self.directories:
            assert dir in dirs

        self.assertEqual(list_files_in_dir(self.directories[0]), [])

    def test_list_valid_directories_exclude_git(self):
        """
        Test: Directories are correct listed with git excluded
        """
        dirs = list_dirs_in_dir(self.directory)
        for dir in dirs:
            assert not dir.endswith('.git')

    def test_list_directories_in_file(self):
        """
        Test: List a file instead of a directory
        """
        with self.assertRaises(AssertionError):
            list_dirs_in_dir(self.files_txt[0])

if __name__ == '__main__':
    unittest.main()
