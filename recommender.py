import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert_json_list(obj):
    """Parse stringified JSON lists for genres and keywords."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except Exception:
        pass
    return L

def get_top_cast(obj):
    """Extract top 3 cast members."""
    L = []
    try:
        for i in ast.literal_eval(obj)[:3]:
            L.append(i['name'])
    except Exception:
        pass
    return L

def get_director(obj):
    """Extract director name from crew."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except Exception:
        pass
    return L

def load_and_preprocess_data(movies_path='data/tmdb_5000_movies.csv', credits_path='data/tmdb_5000_credits.csv'):
    # Load dataset
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge datasets
    movies = movies.merge(credits, on='title')

    # Select relevant features
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    # Handle missing values
    movies.dropna(inplace=True)

    # Clean text structures
    movies['genres'] = movies['genres'].apply(convert_json_list)
    movies['keywords'] = movies['keywords'].apply(convert_json_list)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Collapse multi-word tags
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine features into single tag column
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    return new_df

def compute_similarity_matrix(df):
    """Convert text tags into vectors and calculate cosine similarity."""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie_title, df, similarity, top_n=5):
    """Generate top recommendations for a given movie."""
    movie_title_lower = movie_title.strip().lower()
    matches = df[df['title'].str.lower() == movie_title_lower]
    
    if matches.empty:
        return []

    movie_index = matches.index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommendations = []
    for i in movies_list:
        recommendations.append({
            'movie_id': df.iloc[i[0]].movie_id,
            'title': df.iloc[i[0]].title,
            'score': round(float(i[1]), 3)
        })
    return recommendations

if __name__ == "__main__":
    print("Preprocessing dataset...")
    df = load_and_preprocess_data()
    similarity = compute_similarity_matrix(df)
    
    test_movie = "Inception"
    print(f"\nRecommendations for '{test_movie}':")
    results = recommend(test_movie, df, similarity)
    for idx, rec in enumerate(results, 1):
        print(f"{idx}. {rec['title']} (Score: {rec['score']})")
