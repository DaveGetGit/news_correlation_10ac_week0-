"""
This module defines the NewsDataLoader class for loading news-related datasets.

The NewsDataLoader class provides a method to load a dataset from a given path.
If the dataset has already been loaded, it retrieves the data from memory instead of reloading it.
"""
import argparse
import pandas as pd

class NewsDataLoader:
    '''
    A class that will load news related datasets when provided path.
    '''
    def __init__(self):
        '''
        data: Dictionary to store loaded data.
        '''
        self.data = {}

    def load_data(self, path):
        '''
        Load data from a CSV file at the given path.

        Args:
            path (str): The path to the CSV file.

        Returns:
            DataFrame: The loaded data.
        '''
        if path not in self.data:
            self.data[path] = pd.read_csv(path)
        return self.data[path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export news history')
    parser.add_argument('--zip', help="Name of a zip file to import")
    args = parser.parse_args()
