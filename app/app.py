import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

# --- Load Data and Models ---
@st.cache_resource
def load_data():
    df = pd.read_csv('data/SpotifyFeatures_sample.csv')
    embeddings = np.load('data/song_embeddings_sample.npy')
    index = faiss.read_index('data/faiss_song_sample.index')
    return df, embeddings, index

df, embeddings, index = load_data()

if 'liked_songs' not in st.session_state:
    st.session_state.liked_songs = []
df['Display'] = df['track_name'] + " - " + df['artist_name']


# --- App Title ---
st.title("🎵  Hybrid AI Music Recommendation System ")

# Donut chart (same size as word cloud)
genre_counts = df['genre'].value_counts().head(8)
colors = plt.cm.Set3(range(len(genre_counts)))
fig1, ax1 = plt.subplots(figsize=(7, 7))   # << same figsize as wordcloud
wedges, texts, autotexts = ax1.pie(
    genre_counts,
    labels=genre_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.4)
)
plt.setp(autotexts, size=12, weight="bold")
plt.setp(texts, size=11)
centre_circle = plt.Circle((0, 0), 0.7, fc='white')
fig1.gca().add_artist(centre_circle)

# Word cloud (same size as donut chart)
text = ' '.join(df['artist_name'].dropna())
wordcloud = WordCloud(width=700, height=700, background_color='white', colormap='tab10').generate(text)
fig2, ax2 = plt.subplots(figsize=(7, 7))  # << same figsize as donut chart
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')

# Side by side in Streamlit
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎨 Top 8 Genres in the Catalog")
    st.pyplot(fig1)
with col2:
    st.subheader("✨ Artist Highlights")
    st.pyplot(fig2)

# Mood selection
mood = st.selectbox("Choose a mood/vibe to get song recommendations:", ["-- Select --", "Happy", "Energetic", "Sad", "Chill"])

if mood != "-- Select --":
    # Set up a separate refresh counter for each mood
    refresh_counter_key = f"{mood}_refresh_counter"
    if refresh_counter_key not in st.session_state:
        st.session_state[refresh_counter_key] = 0

    # Show the button, use a unique key for the button itself (unrelated to session state)
    if st.button("🔄 Refresh Songs", key=f"refresh_btn_{mood}"):
        st.session_state[refresh_counter_key] += 1

    # Define mood filters
    if mood == "Happy":
        mood_df = df[df['valence'] > 0.7].copy()
    elif mood == "Energetic":
        mood_df = df[df['danceability'] > 0.7].copy()
    elif mood == "Sad":
        mood_df = df[df['valence'] < 0.3].copy()
    elif mood == "Chill":
        mood_df = df[(df['valence'] > 0.3) & (df['danceability'] < 0.4)].copy()
    else:
        mood_df = pd.DataFrame()

    if not mood_df.empty:
        num_songs = 10
        # Use the refresh counter as a random seed for new shuffles
        np.random.seed(st.session_state[refresh_counter_key])
        mood_display = mood_df.sample(min(num_songs, len(mood_df)))

        # Add YouTube links
        def make_youtube_link(track, artist):
            query = f"{track} {artist}"
            url = f"https://www.youtube.com/results?search_query={'+'.join(str(query).split())}"
            return f"[Play]({url})"
        mood_display['YouTube'] = mood_display.apply(lambda row: make_youtube_link(row['track_name'], row['artist_name']), axis=1)

        # Rename columns
        mood_display.rename(
            columns={
                'track_name': 'Track Name',
                'artist_name': 'Artist',
                'genre': 'Genre',
                'valence': 'Happiness',
                'danceability': 'Dance Factor'
            },
            inplace=True
        )

        # Markdown table for clickable links
        cols = ['Track Name', 'Artist', 'Genre', 'Happiness', 'Dance Factor', 'YouTube']
        header = "| " + " | ".join(cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(mood_display.iloc[i][col]) for col in cols) + " |"
            for i in range(len(mood_display))
        ]
        table_md = "\n".join([header, divider] + rows)
        st.markdown(table_md, unsafe_allow_html=True)
    else:
        st.info("No songs found for this mood in your catalog.")

# --- SONG SEARCH & RECOMMENDATIONS UNIFIED LOGIC ---

matches = pd.DataFrame()
recs = pd.DataFrame()
selected_from_search = None
selected_from_recs = None

song_input = st.text_input("Enter a song name (partial or full):")

if song_input:
    matches = df[df['track_name'].fillna('').str.lower().str.contains(song_input.lower())].copy()
    if matches.empty:
        st.warning("No matching song found.")
    else:
        matches['Display'] = matches['track_name'] + " - " + matches['artist_name']
        selected_from_search = st.selectbox(
            "Songs We Found:",
            matches['Display'],
            key="search_select"
        )
        # Get index of selected song for recs
        query_idx = matches[matches['Display'] == selected_from_search].index[0]

        # --- Recommendation Logic (your code here) ---
        k = 20
        top_n = st.slider("How many recommendations to show?", 3, 10, 5)
        weight_similarity = 0.6
        weight_popularity = 0.15
        weight_valence = 0.1
        weight_danceability = 0.1
        weight_genre = 0.05

        query_vector = embeddings[query_idx].reshape(1, -1)
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        D, I = index.search(query_vector, k + 1)
        candidate_indices = I[0][1:]
        similarity_scores = D[0][1:]

        # Normalize features
        pop_scores = df.iloc[candidate_indices]['popularity'].values
        pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-8)
        valence_scores = df.iloc[candidate_indices]['valence'].values
        valence_norm = (valence_scores - valence_scores.min()) / (valence_scores.max() - valence_scores.min() + 1e-8)
        dance_scores = df.iloc[candidate_indices]['danceability'].values
        dance_norm = (dance_scores - dance_scores.min()) / (dance_scores.max() - dance_scores.min() + 1e-8)
        query_genre = df.iloc[query_idx]['genre']
        genre_match = (df.iloc[candidate_indices]['genre'] == query_genre).astype(float).values

        hybrid_scores = (
            weight_similarity * similarity_scores +
            weight_popularity * pop_norm +
            weight_valence * valence_norm +
            weight_danceability * dance_norm +
            weight_genre * genre_match
        )
        sorted_indices = np.argsort(-hybrid_scores)
        desired_n = top_n
        unique_recommendations = []
        seen = set()
        for idx in sorted_indices:
            song = df.iloc[candidate_indices[idx]]
            key = (song['track_name'], song['artist_name'])
            if key not in seen:
                unique_recommendations.append(idx)
                seen.add(key)
            if len(unique_recommendations) == desired_n:
                break
        top_indices = [candidate_indices[i] for i in unique_recommendations]

        # --- Build Recommendations Table ---
        recs = df.iloc[top_indices][['track_name', 'artist_name', 'genre', 'popularity', 'valence', 'danceability']].copy()
        recs.rename(
            columns={
                'track_name': 'Track Name',
                'artist_name': 'Artist',
                'genre': 'Genre',
                'popularity': 'Popularity',
                'valence': 'Happiness',
                'danceability': 'Dance Factor'
            }, inplace=True
        )
        def make_youtube_link(track, artist):
            query = f"{track} {artist}"
            url = f"https://www.youtube.com/results?search_query={'+'.join(query.split())}"
            return f"[Play]({url})"
        recs['YouTube'] = recs.apply(lambda row: make_youtube_link(row['Track Name'], row['Artist']), axis=1)
        recs = recs.drop_duplicates(subset=['Track Name', 'Artist'])
        # Display as markdown for clickable links
        cols = ['Track Name', 'Artist', 'Genre', 'Popularity', 'Happiness', 'Dance Factor', 'YouTube']
        header = "| " + " | ".join(cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(recs.iloc[i][col]) for col in cols) + " |"
            for i in range(len(recs))
        ]
        table_md = "\n".join([header, divider] + rows)
        st.subheader("Recommended Songs:")
        st.markdown(table_md, unsafe_allow_html=True)

        if not recs.empty and 'Track Name' in recs.columns and 'Artist' in recs.columns:
            recs['Display'] = recs['Track Name'] + " - " + recs['Artist']
            selected_from_recs = st.selectbox(
                "Select a song from recommendations to view details:",
                recs['Display'].tolist(),
                key="recs_select"
            )
else:
    selected_from_search = None
    selected_from_recs = None

# --- SINGLE, UNIFIED SONG DETAILS SECTION ---
selected_song_display = None
if selected_from_recs:
    selected_song_display = selected_from_recs
elif selected_from_search:
    selected_song_display = selected_from_search

if selected_song_display:
    song_row = None
    if not recs.empty and selected_song_display in recs['Display'].values:
        song_row = recs[recs['Display'] == selected_song_display].iloc[0]
    elif not matches.empty and selected_song_display in matches['Display'].values:
        song_row = matches[matches['Display'] == selected_song_display].iloc[0]
        # Rename columns for display consistency
        if 'track_name' in song_row:
            song_row = song_row.rename({
                'track_name': 'Track Name',
                'artist_name': 'Artist',
                'genre': 'Genre',
                'popularity': 'Popularity',
                'valence': 'Happiness',
                'danceability': 'Dance Factor'
            })
    elif selected_song_display in df['Display'].values:
        song_row = df[df['Display'] == selected_song_display].iloc[0]
        if 'track_name' in song_row:
            song_row = song_row.rename({
                'track_name': 'Track Name',
                'artist_name': 'Artist',
                'genre': 'Genre',
                'popularity': 'Popularity',
                'valence': 'Happiness',
                'danceability': 'Dance Factor'
            })
    if song_row is not None:
        cols = ['Track Name', 'Artist', 'Genre', 'Popularity', 'Happiness', 'Dance Factor']
        st.subheader("Song Details:")
        st.dataframe(song_row[cols].to_frame().T)
        yt_url = f"https://www.youtube.com/results?search_query={'+'.join(str(song_row['Track Name'] + ' ' + song_row['Artist']).split())}"
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"[Play on YouTube]({yt_url})", unsafe_allow_html=True)
        with col2:
            like_key = f"like_details_{song_row['Track Name']}_{song_row['Artist']}"
            if st.button("❤️ Like", key=like_key):
                song_tuple = (song_row['Track Name'], song_row['Artist'])
                if song_tuple not in st.session_state.liked_songs:
                    st.session_state.liked_songs.append(song_tuple)
                    st.success(f"Added '{song_row['Track Name']}' by {song_row['Artist']} to your liked songs!")

# --- Liked Songs Section ---
if st.session_state.liked_songs:
    st.subheader("🎶 Your Liked Songs This Session:")
    liked_df = pd.DataFrame(st.session_state.liked_songs, columns=["Track Name", "Artist"])
    st.dataframe(liked_df)
