# 1/3 

import requests
import json
import pandas as pd
import urllib.parse
import folium
import geopandas as gpd
import csv
import time

# Yelp API key
API_KEY = '3c3W1ad7Fk346olp9ISvzsZ7syXj_JTvnwxHJ9YuGWrjTyJ_FhEz6HYIo_eyGOaK8RBmO9xhrd0bush2_Lu2pr5jiWPTXHCIeN8nDZi68U26pmC1dy2U7V-_vgeMZHYx'
# Latitude and longitude for Kadıköy
latitude = 40.9819
longitude = 29.0247

# Query for coffee shops
term = 'coffee'

# URL for Yelp API
url = 'https://api.yelp.com/v3/businesses/search'

# Headers with API key
headers = {
    'Authorization': 'Bearer %s' % API_KEY,
}
parsed_data = []
offset = 0
while True:
    # Parameters
    params = {
        'term': term,
        'latitude': latitude,
        'longitude': longitude,
        'limit': 50,
        'offset': offset,
    }
    
    # Send request and get response
    resp = requests.get(url=url, headers=headers, params=params)
    data = json.loads(resp.text)

    # Get the businesses data
    businesses = data.get('businesses', [])
    if not businesses:
        break

    # Parse the data
    for business in businesses:
        business_name = business['name']
        business_lat = business['coordinates']['latitude']
        business_lng = business['coordinates']['longitude']
        business_category = business['categories'][0]['title'] if business['categories'] else None
        business_rating = business['rating']
        business_review_count = business['review_count']
        parsed_data.append([business_name, business_lat, business_lng, business_category, business_rating, business_review_count])

    # Increase the offset by the number of results in this batch
    offset += len(businesses)

    # As to avoid hitting Yelp's rate limits, it's a good idea to wait some time before making the next request
    time.sleep(1)
    
# Convert to pandas DataFrame
import pandas as pd
dfcafes = pd.DataFrame(parsed_data, columns=['Name', 'Latitude', 'Longitude', 'Category', 'Rating', 'ReviewCount'])

# Drop any row with missing data
dfcafes = dfcafes.dropna()

