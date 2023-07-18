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
API_KEY = 'API_KEY HERE'

# Geographic coordinates for Istanbul
latitude = 41.0151
longitude = 28.9795

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

    resp = requests.get(url=url, headers=headers, params=params)
    data = json.loads(resp.text)

    # Getting the businesses data
    businesses = data.get('businesses', [])
    if not businesses:
        break

    # Parsing the data
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

    # To avoid hitting Yelp's 50 result/query limit
    time.sleep(1)
    

import pandas as pd
dfcafes = pd.DataFrame(parsed_data, columns=['Name', 'Latitude', 'Longitude', 'Category', 'Rating', 'ReviewCount'])


dfcafes = dfcafes.dropna()

