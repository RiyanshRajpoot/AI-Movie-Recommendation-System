import streamlit as st
from recommender import load_and_preprocess_data, compute_similarity_matrix, recommend

st.set_page_config(page_title="AI Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 AI Movie Recommendation System")
st.write("Find movies similar to your favorites using Natural Language Processing and Cosine Similarity.")

@st.cache_data
def get_model_data():
    df = load_and_preprocess_data()
    similarity = compute_similarity_matrix(df)
    return df, similarity

with st.spinner("Loading movie engine..."):
    df, similarity = get_model_data()

movie_list = df['title'].values
selected_movie = st.selectbox("Select or search a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, df, similarity, top_n=5)
    
    if recommendations:
        st.subheader(f"Top 5 Recommendations for '{selected_movie}':")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(recommendations):
                rec = recommendations[idx]
                with col:
                    st.success(f"**{rec['title']}**")
                    st.caption(f"Similarity Score: {rec['score']}")
    else:
        st.error("Movie not found in database. Try another title.")
