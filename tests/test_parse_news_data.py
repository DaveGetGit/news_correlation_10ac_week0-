"""
Unit tests for the NewsDataLoader class and the find_high_traffic_websites function.
"""

import sys
import os
import unittest
import pandas as pd
from src.utils import find_high_traffic_websites
from src.loader import NewsDataLoader

# Ensure the parent directory is in the system path
parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class TestNewsDataLoader(unittest.TestCase):
    """
    Unit Test class for testing NewsDataLoader class and find_high_traffic_websites function.
    """
    def setUp(self):
        """
        Set up method for initializing common test variables.
        """
        self.data_loader = NewsDataLoader()
        self.traffic_dataset_path = '../data/traffic.csv'

    def test_load_data(self):
        """
        Test case for verifying the load_data method of the NewsDataLoader class.
        """
        traffic_data = self.data_loader.load_data(self.traffic_dataset_path)
        self.assertIsInstance(traffic_data, pd.DataFrame)
        self.assertFalse(traffic_data.empty)

    def test_find_high_traffic_websites(self):
        """
        Test case for verifying the find_high_traffic_websites function.
        """
        traffic_data = self.data_loader.load_data(self.traffic_dataset_path)
        top_traffic_websites = find_high_traffic_websites(traffic_data)
        self.assertIsInstance(top_traffic_websites, pd.Series)
        self.assertFalse(top_traffic_websites.empty)

if __name__ == '__main__':
    unittest.main()
