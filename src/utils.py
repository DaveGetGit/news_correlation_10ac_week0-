"""
This module contains functions and classes for analyzing news data. 

It includes methods for:
- Finding top websites
- Identifying high traffic websites
- Determining countries with most media outlets
- Discovering popular articles based on countries mentioned in their content

Additionally, it provides functionality for calculating sentiment counts and 
distribution for each website.
"""

from collections import Counter
from multiprocessing import Pool
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from tqdm import tqdm
from rake_nltk import Rake

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np

def find_top_websites(news_data, url_column='url', top_count=10):
    """
    Get the top [top_count] websites with the highest article counts.
    """
    news_data['domain'] = news_data[url_column].apply(lambda x: x.split('/')[2])

    # Count occurrences of each domain
    domain_counts = news_data['domain'].value_counts()

    top_domains = domain_counts.head(top_count)
    return top_domains

def find_high_traffic_websites(news_data, top_count=10):
    """
    Get websites with high reference IPs (assuming the IPs are the number of traffic).
    """
    traffic_per_domain = news_data.groupby(['Domain'])['RefIPs'].sum()
    traffic_per_domain = traffic_per_domain.sort_values(ascending=False)
    return traffic_per_domain.head(top_count)

def find_countries_with_most_media(news_data, top_count=10):
    """
    Get the top countries with the most media outlets.
    """
    media_per_country = news_data['Country'].value_counts()
    media_per_country = media_per_country.sort_values(ascending=False)
    return media_per_country.head(top_count)

def download_nltk_resources():
    """
    Download required NLTK packages.
    """
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

def extract_countries_from_article_content(article):
    """
    Extract countries from the content of each article using NLTK.
    """
    row = article
    text = row['content']
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)
    countries = [
        chunk[0] for chunk in list(named_entities)
        if hasattr(chunk, 'label') and chunk.label() == 'GPE'
    ]
    return countries


def find_popular_articles(articles_data):
    """
    Find the most popular articles based on countries mentioned in their content.
    """
    print('Downloading NLTK resources ...')
    download_nltk_resources()
    print('Finished downloading resources ...')
    print('Loading data ...')
    df = articles_data
    print('Starting processing; this might take a while ...')

    max_rows = len(df)
    print(f'Max rows: {max_rows}')

    with Pool() as pool:
        results = []
        for countries in tqdm(pool.imap(extract_countries_from_article_content
                                        , df.iterrows()), total=len(df)):
            results.append(countries)
            if len(results) >= max_rows:
                print("Maximum number of rows processed. Stopping pool.")
                break

    print('Done processing!')
    all_countries = [country for countries in results for country in countries]
    country_counts = Counter(all_countries)
    print(country_counts.most_common(3))
    return country_counts.most_common(10)

def website_sentiment_counts(data):
    """
    Calculate sentiment counts for each website.
    """
    sentiment_counts = data.groupby(['source_name', 'title_sentiment']).size().unstack(fill_value=0)
    return sentiment_counts

def website_sentiment_distribution(data):
    """
    Calculate sentiment distribution for each website.
    """
    sentiment_counts = data.groupby(['source_name',
                                     'title_sentiment']).size().unstack(fill_value=0)
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)
    sentiment_counts['Mean'] = sentiment_counts[['Positive',
                                                 'Neutral', 'Negative']].mean(axis=1)
    sentiment_counts['Median'] = sentiment_counts[['Positive',
                                                    'Neutral', 'Negative']].median(axis=1)
    print("Sentiment counts with mean and median:")
    print(sentiment_counts)
    return sentiment_counts





def keyword_extraction_and_analysis(news_data):
    """
    Extracts keywords from the titles and contents of news articles and analyzes their similarity.

    Args:
    news_data (DataFrame): DataFrame containing news articles with 'title' and 'content' columns.

    Returns:
    tuple: A tuple containing lists of top keywords from titles and contents, and a list of similarity scores.
    """

    stop_words = set(stopwords.words('english'))

    vectorizer = TfidfVectorizer(max_features=10, min_df=1)  

    title_keywords_list = []
    content_keywords_list = []
    similarity_list = []

    MIN_WORDS_THRESHOLD = 5  
    article_count = 0 
    for index, row in news_data.iterrows():

        if(article_count == 101): break

        title_text = row['title']

        processed_title = [word.lower() for word in word_tokenize(title_text) if word.lower() not in stop_words and word.isalpha()]
        content_text = row['content']

        processed_content = [word.lower() for word in word_tokenize(content_text) if word.lower() not in stop_words and word.isalpha()]

        if len(processed_title) < MIN_WORDS_THRESHOLD and len(processed_content) < MIN_WORDS_THRESHOLD:
            continue

        combined_text = ' '.join(processed_title + processed_content)

        vectorizer.fit([combined_text])

        tfidf_scores = vectorizer.transform([combined_text]).toarray()[0]

        feature_names = vectorizer.get_feature_names_out()

        top_keywords_title = [keyword for keyword, _ in sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:5]]
        top_keywords_content = [keyword for keyword, _ in sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[5:10]]

        if(len(tfidf_scores) >= 10):

            title_keywords_list.append(top_keywords_title)
            content_keywords_list.append(top_keywords_content)

            title_tfidf_scores = tfidf_scores[:5]
            content_tfidf_scores = tfidf_scores[5:10]

            similarity = 1 - cosine(title_tfidf_scores, content_tfidf_scores)

            similarity_list.append(similarity)
            article_count = article_count+1
        else:
            print('Cannot calculate similarity on unbalanced keywords')

    return title_keywords_list, content_keywords_list, similarity_list


own_categories = {
    0: 'Breaking News',
    1: 'Politics',
    2: 'World News',
    3: 'Business/Finance',
    4: 'Technology',
    5: 'Science',
    6: 'Health',
    7: 'Entertainment',
    8: 'Sports',
    9: 'Environment',
    10: 'Crime',
    11: 'Education',
    12: 'Weather'
}


def clean_text(text):
    """
    Removes HTML tags and special characters from text.

    Args:
    text (str): Input text.

    Returns:
    str: Cleaned text.
    """
    clean_text = re.sub('<.*?>','',text)
    clean_text = re.sub(r'[^\w\s]','',text)
    return clean_text

def preprocess_text(text):
    """
    Preprocesses text by removing stop words and lemmatizing words.

    Args:
    text (str): Input text.

    Returns:
    str: Preprocessed text.
    """
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

