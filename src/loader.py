import argparse
import pandas as pd

class NewsDataLoader:
    '''
    A class that loads news-related datasets when provided a path.
    '''
    def __init__(self):
        '''
        data: Dictionary to store loaded data
        '''
        self.loaded_data = {}
    
    def load_data(self, path):
        '''
        Load data from the provided path and store it in memory.
        '''
        if path not in self.loaded_data:
            self.loaded_data[path] = pd.read_csv(path)
        return self.loaded_data[path]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export news history')
    parser.add_argument('--zip_file', help="Name of a zip file to import")
    args = parser.parse_args()
