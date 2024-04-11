"""
This module contains tests for the NewsDataLoader class and the find_top_websites function.
"""
import sys
import os
import pandas as pd
from src.utils import find_top_websites
from src.loader import NewsDataLoader

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_load_data():
    """
    Test the load_data method of the NewsDataLoader class.
    """
    # Initialize the data loader
    data_loader = NewsDataLoader()

    # Load the news data
    news_dataset_path = '../data/data.csv'
    news_data = data_loader.load_data(news_dataset_path)

    assert isinstance(news_data, pd.DataFrame), "The data should be a pandas DataFrame"
    assert not news_data.empty, "The DataFrame should not be empty"

def test_find_top_websites():
    """
    Test the find_top_websites function.
    """
    # Initialize the data loader
    data_loader = NewsDataLoader()

    # Load the news data
    news_dataset_path = '../data/data.csv'
    news_data = data_loader.load_data(news_dataset_path)

    # Find the top websites
    top_news_websites = find_top_websites(news_data)

    assert isinstance(top_news_websites, pd.Series), "The output should be a pandas Series"
    assert not top_news_websites.empty, "The Series should not be empty"
