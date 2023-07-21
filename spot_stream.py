#!/usr/bin/env python
# coding: utf-8

# In[22]:


import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth


# In[23]:


scope = 'user-read-recently-played'


# In[24]:


st.title("My Spotify Journey")


# In[25]:


client_id = st.text_input("Client ID")
client_secret = st.text_input("Client Secret", type="password")
redirect_uri = st.text_input("Redirect URI")


# In[28]:


if st.button("Get My Recent Tracks"):
    
    if client_id and client_secret and redirect_uri:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                       client_secret=client_secret,
                                                       redirect_uri=redirect_uri,
                                                       scope=scope))
        
        results = sp.current_user_recently_played(limit=50)
        data=[]
        
        for idx, item in enumerate(results['items']):
            artists = [artist['name'] for artist in item['track']['artists']]
            track = item['track']['name']
            played_at = item['played_at']
            data.append({"Track": track, "Artist":','.join(artists), "Played at": played_at})
            
        daf = pd.DataFrame(data)
        st.dataframe(daf)
            
            #st.write(f"Track: {item['track']['name']}, Artist(s): {', '.join(artists)}, Played at: {item['played_at']}")
        
    else:
        st.warning("Please input all Spotify API credentials!")
            
            


# In[ ]:




