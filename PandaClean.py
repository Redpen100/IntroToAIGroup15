# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:30:12 2022

@author: spide
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

##Create datafram and drop un-necessary columns
df = pd.read_csv('Spotify-Dupes.csv',index_col=0)
df.drop(['track_id', 'album_name'], axis = 1, inplace = True)

##Create copy of db
df2 = df

##Encodes all strings to numerics
le = LabelEncoder()
df2['track_name'] = le.fit_transform(df['track_name'])
df2['artists'] = le.fit_transform(df2['artists'])
df2['track_genre'] = le.fit_transform(df2['track_genre'])
df2['explicit'] = le.fit_transform(df2['explicit'])

##Groups data by trackname and artist
df2 = df2.groupby(["track_name", "artists"]).agg({"popularity":lambda x: x.iloc[0],"duration_ms":lambda x: x.iloc[0],"explicit":lambda x: x.iloc[0],"danceability":lambda x: x.iloc[0],
"energy":lambda x: x.iloc[0],"key":lambda x: x.iloc[0],"loudness":lambda x: x.iloc[0],"mode":lambda x: x.iloc[0],"speechiness":lambda x: x.iloc[0],
"acousticness":lambda x: x.iloc[0],"instrumentalness":lambda x: x.iloc[0],"liveness":lambda x: x.iloc[0], "track_genre": list})

df2 = df2.dropna()
##Exports to csv for checking
df2.to_csv('TestDataExport.csv')
