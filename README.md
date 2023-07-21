# my_spoti

## Personal Spotify Analysis 
This project uses the Spotify and Genius APIs, along with a BERTweet model for sentiment analysis, to analyze the sentiment of the lyrics of recently played songs on Spotify for creating a Mood Progression on a chart. 

## Interactive Website for Spotify Analysis
Spotify's Web API imposes a limit on the number of results returned by certain endpoints. This limit is part of Spotify's API design and cannot be overridden. For example, when fetching a user's recently played tracks, the maximum limit is set to 50. This means that we cannot retrieve more than 50 recently played tracks in a single API call.

On the other hand, Streamlit applications are focusing primarily on the application's main thread. Streamlit does not directly support pop-up windows or redirecting to external URLs within the same application window. This can cause a challenge when integrating with APIs like Spotify's, which often require user authentication via a redirect URI in a separate browser tab or window. So I was only able to run in my local.

###Here is the provided website: https://yourspotistatis.streamlit.app/

![image](https://github.com/nazlicanto/my_spoti/assets/117021695/a63f0f51-2894-4f39-b8f5-bc2d713bb48c)

