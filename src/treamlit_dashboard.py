import streamlit as streamlit_interface
from src.utils import find_top_websites, find_popular_articles, find_high_traffic_websites, find_countries_with_most_media
from src.loader import NewsDataLoader

data_loader = NewsDataLoader()

def display_main_dashboard():
    streamlit_interface.title('News Analysis Dashboard')
    streamlit_interface.sidebar.title('Options')

    data_frame = data_loader.load_data('../data/data.csv/traffic.csv')

    analysis_options = {
        'Top Websites by Article Count': find_top_websites,
        'High Traffic Websites': find_high_traffic_websites,
        'Countries with Most Media Outlets': find_countries_with_most_media,
        'Popular Articles by Country': find_popular_articles
    }

    selected_analysis = streamlit_interface.sidebar.selectbox('Select Function', list(analysis_options.keys()))

    top_count = 10

    if selected_analysis == 'Popular Articles by Country':
        streamlit_interface.info("Loading....")
        popular_countries_data = data_frame[['index_column', 'content']]  # Replace 'index_column' with your actual index column name
        top_countries = find_popular_articles(popular_countries_data)
        streamlit_interface.write("Top 10 Countries with Most Articles:")
        streamlit_interface.write(top_countries)
    else:
        if selected_analysis == 'Top Websites by Article Count':
            top_websites = find_top_websites(data_frame, top_count=top_count)
            streamlit_interface.write("Top Websites by Article Count:")
            streamlit_interface.write(top_websites)

        elif selected_analysis == 'High Traffic Websites':
            high_traffic_websites = find_high_traffic_websites(data_frame, top_count=top_count)
            streamlit_interface.write("High Traffic Websites:")
            streamlit_interface.write(high_traffic_websites)

        elif selected_analysis == 'Countries with Most Media Outlets':
            top_media_countries = find_countries_with_most_media(data_frame, top_count=top_count)
            streamlit_interface.write("Countries with Most Media Outlets:")
            streamlit_interface.write(top_media_countries)

if __name__ == "__main__":
    display_main_dashboard()