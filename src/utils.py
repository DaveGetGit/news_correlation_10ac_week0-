import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

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

count = 0

def extract_countries_from_article_content(article):
    """
    Extract countries from the content of each article using NLTK.
    """
    index, row = article
    text = row['content']
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)
    countries = [chunk[0] for chunk in list(named_entities) if hasattr(chunk, 'label') and chunk.label() == 'GPE']
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
    processed_count = 0

    with Pool() as pool:
        results = []
        for countries in tqdm(pool.imap(extract_countries_from_article_content, df.iterrows()), total=len(df)):
            results.append(countries)
            processed_count += 1
            if processed_count >= max_rows:
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
    sentiment_counts = data.groupby(['source_name', 'title_sentiment']).size().unstack(fill_value=0)
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)
    sentiment_counts['Mean'] = sentiment_counts[['Positive', 'Neutral', 'Negative']].mean(axis=1)
    sentiment_counts['Median'] = sentiment_counts[['Positive', 'Neutral', 'Negative']].median(axis=1)
    print("Sentiment counts with mean and median:")
    print(sentiment_counts)
    return sentiment_counts
