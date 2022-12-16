# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:30:12 2022

@author: spide
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Spotify-Dupes.csv',index_col=0)
df.drop(['track_id', 'album_name'], axis = 1, inplace = True)

# Use the `str.split` method to split the `artists` column into a list of artists
df["artists"] = df["artists"].str.split(';')

# Convert the `explicit` column to a boolean dtype
df['explicit'] = df['explicit'].astype(bool)

# Use the `apply` method to apply a function to each element of the `track_genre` column
# The function maps the values of the `track_genre` column to integers using a dictionary
track_genre_dict = {genre: i for i, genre in enumerate(df['track_genre'].unique())}
df['track_genre'] = df['track_genre'].apply(lambda x: track_genre_dict[x])

# Use the `groupby` method to group the data by `track_name`
df2 = df.groupby(["track_name"]).agg({"artists":lambda x: x.iloc[0],
                                      "popularity":lambda x: x.iloc[0],
                                      "duration_ms":lambda x: x.iloc[0],
                                      "explicit":lambda x: x.iloc[0],
                                      "danceability":lambda x: x.iloc[0],
                                      "energy":lambda x: x.iloc[0],
                                      "key":lambda x: x.iloc[0],
                                      "loudness":lambda x: x.iloc[0],
                                      "mode":lambda x: x.iloc[0],
                                      "speechiness":lambda x: x.iloc[0],
                                      "acousticness":lambda x: x.iloc[0],
                                      "instrumentalness":lambda x: x.iloc[0],
                                      "liveness":lambda x: x.iloc[0],
                                      "track_genre": list})

# Use the `apply` method to apply a function to each element of the `artists` column
# The function maps the values of the `artists` column to integers using a dictionary
all_artists = df2["artists"].apply(pd.Series).stack().unique()
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df2['artists'] = df2['artists'].apply(lambda x: [artist_dict[a] for a in x])

df2 = df2.dropna()
##print(df2.head())
# Print the head of the resulting DataFrame
df2.to_csv('TestDataExport.csv')
