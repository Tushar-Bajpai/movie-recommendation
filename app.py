import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re

# Load Dataset
data = pd.read_csv('tmdb_5000_movies.csv')

# Data Preprocessing
data['combined_features'] = (data['genres'] + " " + 
                             data['release_date'].fillna('') + " " + 
                             data['keywords'].fillna('') + " " + 
                             data['original_title'].fillna('') + " " + 
                             data['overview'].fillna(''))

# Normalize and prepare data for genre-based recommendations
data['title_normalized'] = data['original_title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
data['genres_normalized'] = data['genres'].str.lower()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(data['combined_features'])

# Cosine Similarity Matrix
similarity_matrix = cosine_similarity(feature_vectors)

# Recommendation by Similarity
def recommend(movie_name, top_n=5):
    """
    Recommend movies based on similarity to the input movie.
    """
    movie_name = movie_name.lower()
    try:
        movie_index = data[data['title_normalized'] == re.sub(r'[^\w\s]', '', movie_name)].index[0]
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_movies = [data['original_title'][i[0]] for i in sorted_movies[1:top_n + 1]]
        return recommended_movies
    except IndexError:
        return "Movie not found in the dataset."

# Recommendation by Genre
def recommend_based_on_name_and_genre(movie_name, top_n=5):
    """
    Recommend movies based on genre similarity to the input movie.
    """
    movie_name = movie_name.lower()
    try:
        movie_name_normalized = re.sub(r'[^\w\s]', '', movie_name)
        movie_index = data[data['title_normalized'] == movie_name_normalized].index[0]
        movie_genre = data['genres_normalized'][movie_index]
        genre_movies = data[data['genres_normalized'] == movie_genre]
        genre_movies = genre_movies[genre_movies['title_normalized'] != movie_name_normalized]
        recommended_movies = genre_movies.head(top_n)
        return recommended_movies[['original_title', 'genres']]
    except IndexError:
        return pd.DataFrame(columns=['original_title', 'genres'])

# Streamlit UI
st.title("Movie Recommendation System")

st.markdown("""
    <style>
    body {
        background-color: #f0f0f0;
        color: black;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        border-radius: 8px;
        padding: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Input for movie name
movie_name = st.text_input("Enter a movie name:")

# Button to get recommendations
if st.button("Get Recommendations"):
    if movie_name:
        recommendations = recommend(movie_name)
        genre_based_recommendations = recommend_based_on_name_and_genre(movie_name)

        if isinstance(recommendations, list):
            st.write(f"Movies similar to '{movie_name}':")
            for movie in recommendations:
                st.write(f"- {movie}")
        else:
            st.write(recommendations)

        if not genre_based_recommendations.empty:
            st.write(f"Movies from the same genre as '{movie_name}':")
            for index, row in genre_based_recommendations.iterrows():
                st.write(f"- {row['original_title']}")
        else:
            st.write("No genre-based recommendations found.")
    else:
        st.write("Please enter a movie name.")
