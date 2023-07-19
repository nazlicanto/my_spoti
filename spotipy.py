import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lyricsgenius
from textblob import TextBlob
import time
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
import plotly.express as px


import os 


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='...',
                                               client_secret='...',
                                               redirect_uri='...',
                                               scope='user-top-read'))


genre_artist_dict = {}


for item in results['items']:

    for genre in item['genres']:

        if genre in genre_artist_dict:
            genre_artist_dict[genre].append(item['name'])

        else:
            genre_artist_dict[genre] = [item['name']]


for genre, artists in genre_artist_dict.items():
    print(genre, artists)


genres = genre_artist_dict


genre_dict = {
    "techno": [],
    "rock": [],
    "pop": [],
    "jazz": [],
    "folk": [],
    "metal": [],
    "psych": [],
    "indie": [],
    "blues": [],
    "soul": [],
    "rap": [],
    "electronic": [],
    "hip hop": [],
    "house": [],
    "punk": [],
    "alternative": [],
    "new rave": [],
    "thrash": [],
    "new wave": [],
    
    
    "other": []
}


for genre in genres:

    if "techno" in genre:
        genre_dict["techno"].append(genre)
    elif "rock" in genre:
        genre_dict["rock"].append(genre)
    elif "pop" in genre:
        genre_dict["pop"].append(genre)
    elif "jazz" in genre:
        genre_dict["jazz"].append(genre)
    elif "folk" in genre:
        genre_dict["folk"].append(genre)
    elif "metal" in genre:
        genre_dict["metal"].append(genre)
    elif "psych" in genre:
        genre_dict["psych"].append(genre)
    elif "indie" in genre:
        genre_dict["indie"].append(genre)
    elif "blues" in genre:
        genre_dict["blues"].append(genre)
    elif "soul" in genre:
        genre_dict["soul"].append(genre)
    elif "rap" in genre:
        genre_dict["rap"].append(genre)
    elif "electronic" in genre:
        genre_dict["electronic"].append(genre)
    elif "hip hop" in genre:
        genre_dict["hip hop"].append(genre)
    elif "house" in genre:
        genre_dict["house"].append(genre)
    elif "punk" in genre:
        genre_dict["punk"].append(genre)
    elif "alternative" in genre:
        genre_dict["alternative"].append(genre)
    elif "new rave" in genre:
        genre_dict["new rave"].append(genre)
    elif "thrash" in genre:
        genre_dict["thrash"].append(genre)
    elif "wave" in genre:
        genre_dict["new wave"].append(genre)

        

    else:
        genre_dict["other"].append(genre)


for key, value in genre_dict.items():
    print(key, value)


def categorize_artists(genre_artist_dict, genre_keywords):

    genre_dict = {keyword: [] for keyword in genre_keywords}
    genre_dict['other'] = []


    for genre, artists in genre_artist_dict.items():
        # Find a matching genre keyword
        matching_keyword = next((keyword for keyword in genre_keywords if keyword in genre), 'other')
        genre_dict[matching_keyword].extend(artists)

    return genre_dict


genre_keywords = [
    'techno', 'rock', 'pop', 'jazz', 'folk', 'metal', 'psych', 'indie', 'blues',
    'soul', 'rap', 'electronic', 'hip hop', 'house', 'punk', 'alternative',
    'new rave', 'thrash', 'new wave'
]


genre_dict = categorize_artists(genre_artist_dict, genre_keywords)


for key, value in genre_dict.items():
    print(key, value)


df = pd.DataFrame({
    'Genre': list(genre_dict.keys()),
    'Count': [len(v) for v in genre_dict.values()],
    'Artists': [', '.join(v) for v in genre_dict.values()]
})


fig = px.line(df, x='Genre', y='Count', title='Number of Artists(Medium-Term) per Genre')
fig.update_traces(mode='lines+markers', line=dict(color='green', width=4))


fig.update_traces(hovertemplate='Genre: %{x}<br>Count: %{y}<br>Artists: %{text}', text=df['Artists'])


fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_showgrid=True, 
    yaxis_showgrid=True, 
    xaxis_gridcolor='white', 
    yaxis_gridcolor='white'
)


###RECOMMENDATIONS


top_genres = [genre for genre, _ in sorted(genre_dict.items(), key=lambda item: len(item[1]), reverse=True)[:5]]

recommendations = sp.recommendations(seed_genres=top_genres, limit=20)

for track in recommendations['tracks']:
    print(track['name'], '-', track['artists'][0]['name'])


###MOOD CHART

os.environ['TRANSFORMERS_CACHE'] = 'C://Users//nazli//anaconda3//Lib//site-packages'

model_name = "finiteautomata/bertweet-base-sentiment-analysis"
cache_dir = 'C://Users//nazli//anaconda3//Lib//site-packages'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='...',
                                               client_secret='...',
                                               redirect_uri='...',
                                               scope='user-read-recently-played'))

results = sp.current_user_recently_played(limit=50)



for idx, item in enumerate(results['items']):
    artists = [artist['name'] for artist in item['track']['artists']]
    print(f"Index: {idx}, Track: {item['track']['name']}, Artist(s): {', '.join(artists)}, Played at: {item['played_at']}")


genius = lyricsgenius.Genius('APIKEY')

data = []

for idx, item in enumerate(results['items']):

    track_name = item['track']['name']
    artists = ', '.join([artist['name'] for artist in item['track']['artists']])
    played_at = item['played_at']
    
    # Retry up to 3 times
    for _ in range(3):
        try:
            song = genius.search_song(track_name, artists)
            if song is not None: 
                lyrics = song.lyrics
            else: 
                lyrics = ""
                continue
            break
        except Timeout:
            print("Request timed out. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            
    song = genius.search_song(track_name, artists)
    
    if song is not None: 
        lyrics = song.lyrics
    else: 
        lyrics = ""
        continue

    # Split lyrics into chunks of 128 tokens each
    lyrics_chunks = [lyrics[i:i+128] for i in range(0, len(lyrics), 128)]

    sentiment_scores = []



    inputs = tokenizer(lyrics[:512], return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        sentiment_scores = torch.sigmoid(outputs.logits).numpy().flatten()

    # Average the sentiment scores
    avg_sentiment_score = sentiment_scores.mean()

    data.append({
        'track_name': track_name,
        'artists': artists,
        'lyrics': lyrics,
        'sentiment_score': avg_sentiment_score,
        'played_at': played_at 
    })

# Create a line plot with spline interpolation
fig = go.Figure(data=go.Scatter(x=sentiment_df.index, y=sentiment_df['sentiment_score'], mode='lines', line_shape='spline', line=dict(color='black')))

# Set plot title and labels
fig.update_layout(
    title='Sentiment Score Over Time',
    xaxis_title='Index',
    yaxis_title='Sentiment Score',
    plot_bgcolor='beige'  # Change the background color to beige
)

# Show the plot
fig.show()
