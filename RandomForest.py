# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:30:12 2022

@author: spide
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score



df = pd.read_csv('Spotify-Dupes.csv',index_col=0)
df.drop(['track_id', 'album_name'], axis = 1, inplace = True)

# Use the `str.split` method to split the `artists` column into a list of artists
df["artists"] = df["artists"].str.split(';')

# Convert the `explicit` column to a boolean dtype
df['explicit'] = df['explicit'].astype(bool)

# Use the `groupby` method to group the data by `track_name`
df = df.groupby(["track_name"]).agg({"artists":lambda x: x.iloc[0],
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


#df= df[~df['track_genre'].isin(["j-pop"])]
df = df[~df['track_genre'].apply(lambda x: 'j-pop' in x or 'anime' in x or 'black-metal' in x or 'bluegrass' in x or 'brazil' in x or 'cantopop' in x
                                  or 'french' in x or 'german' in x or 'indian' in x or 'iranian' in x or 'j-dance' in x or 'j-idol' in x
                                  or 'j-rock' in x or'k-pop' in x or 'malay' in x or 'mandopop' in x or 'swedish' in x or 'turkish' in x or 'world-music' in x)]


df = df.sample(20000)
# Use the `apply` method to apply a function to each element of the `track_genre` column
# The function maps the values of the `track_genre` column to integers using a dictionary
all_genres = df["track_genre"].apply(pd.Series).stack().unique()
track_genre_dict = {genre: i for i, genre in enumerate(all_genres)}
df['track_genre'] = df['track_genre'].apply(lambda x: [track_genre_dict[a] for a in x])


all_artists = df["artists"].apply(pd.Series).stack().unique()
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df['artists'] = df['artists'].apply(lambda x: [artist_dict[a] for a in x])

df = df.dropna()



mlb = MultiLabelBinarizer()
artists_bin = mlb.fit_transform(df['artists'])
track_genre_bin = mlb.fit_transform(df['track_genre'])
print(df)
#print(df2)
#print(df2.columns)

#df2.columns = ["track_name","artists","popularity","duration_ms","explicit","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","track_genre"]
#for label in df2.columns:
    #df2[label] = LabelEncoder().fit(df2[label]).transform(df2[label])

# Create our X and y data    
result = []
for x in df.columns:
    if (x != 'track_genre' and x!='artists'):
        result.append(x)

X = df[result].values
#track_genre_bin.to_csv('TestDataExport.csv')
y = track_genre_bin

# Concatenate the transformed `artists` and `track_genre` columns to the input data
X = np.concatenate((X, artists_bin), axis=1)

# Initialize the MultiOutputClassifier
clf = MultiOutputClassifier(RandomForestClassifier(n_jobs = 4, n_estimators = 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)
# Use the `apply` method to apply a function to each element of the `artists` column
# The function maps the values of the `artists` column to integers using a dictionary

##print(df2.head())
# Print the head of the resulting DataFrame
#y.to_csv('TestDataExport.csv')

#print(X[:5])

#Instantiate the model with 10 trees and entropy as splitting criteria
##Random_Forest_model = RandomForestClassifier(n_estimators=10,criterion="entropy")

#Training/testing split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

#Train the model
##Random_Forest_model.fit(X_train, y_train)

#make predictions
##y_pred = Random_Forest_model.predict(X_test)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

print(y_pred[:5])
print(y_test[:5])

#Calculate accuracy metric
accuracy = accuracy_score(y_pred, y_test)
mprecision = precision_score(y_test, y_pred, average='micro')
macprecision = precision_score(y_test, y_pred, average='macro')
wprecision = precision_score(y_test, y_pred, average='weighted')
sprecision = precision_score(y_test, y_pred, average='samples')
f=f1_score(y_test, y_pred, average='micro')


print('The accuracy is: ',accuracy*100,'%')
print('The Micro Precision is: ',mprecision*100,'%')
print('The Macro Precision is: ',macprecision*100,'%')
print('The Weight Precision is: ',wprecision*100,'%')
print('The Sample Precision is: ',sprecision*100,'%')
print("f1 =  ", f*100 , "%")

