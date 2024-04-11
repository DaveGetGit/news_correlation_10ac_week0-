"""
Unit tests for the NewsDataLoader class and the find_countries_with_most_media function.
"""

import sys
import os
import unittest
import pandas as pd
from src.utils import find_countries_with_most_media
from src.loader import NewsDataLoader

# Ensure the parent directory is in the system path
parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class TestNewsDataLoader(unittest.TestCase):
    """
    Unit Test class for testing NewsDataLoader class and find_countries_with_most_media function.
    """
    def setUp(self):
        """
        Set up method for initializing common test variables.
        """
        self.data_loader = NewsDataLoader()
        self.domain_location_dataset_path = '../data/domains_location.csv'

    def test_load_data(self):
        """
        Test case for verifying the load_data method of the NewsDataLoader class.
        """
        domain_location_data = self.data_loader.load_data(self.domain_location_dataset_path)
        self.assertIsInstance(domain_location_data, pd.DataFrame)
        self.assertFalse(domain_location_data.empty)

    def test_find_countries_with_most_media(self):
        """
        Test case for verifying the find_countries_with_most_media function.
        """
        domain_location_data = self.data_loader.load_data(self.domain_location_dataset_path)
        countries_with_most_media = find_countries_with_most_media(domain_location_data)
        self.assertIsInstance(countries_with_most_media, pd.Series)
        self.assertFalse(countries_with_most_media.empty)

if __name__ == '__main__':
    unittest.main()
