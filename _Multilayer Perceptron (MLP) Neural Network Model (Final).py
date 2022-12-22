#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#All necessary imports
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



# Read the data frame
df = pd.read_csv('/Users/kanso/OneDrive/Desktop/AI Coursework/dataset.csv',index_col=0)
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



df = df.sample(20000) #Change Sample Value

all_genres = df["track_genre"].apply(pd.Series).stack().unique()
track_genre_dict = {genre: i for i, genre in enumerate(all_genres)}
df['track_genre'] = df['track_genre'].apply(lambda x: [track_genre_dict[a] for a in x])
df['track_genre'] = df['track_genre'].apply(set)
df['track_genre'] = df['track_genre'].apply(list)


all_artists = df["artists"].apply(pd.Series).stack().unique()
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df['artists'] = df['artists'].apply(lambda x: [artist_dict[a] for a in x])
df['artists'] = df['artists'].apply(set)
df['artists'] = df['artists'].apply(list)

df = df.dropna()

# Use the MultiLabelBinarizer to one-hot encode the categorical columns
mlb = MultiLabelBinarizer()
df = df
artists_bin = mlb.fit_transform(df['artists'])
track_genre_bin = mlb.fit_transform(df['track_genre'])
print(df)
#print(df)

# Create our X and y data
result = []
for x in df.columns:
    if (x != 'track_genre' and x!='artists'):
        result.append(x)

X = df[result].values
#track_genre_bin.to_csv('TestDataExport.csv')
y = track_genre_bin

# Concatenate the transformed `artists` and `track_genre` columns to the input data
X = np.concatenate((X, artists_bin), axis=1) #With and Without

# Initialize the MultiOutputClassifier with the MLPClassifier

clf = MultiOutputClassifier(MLPClassifier(random_state=1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Change the test size 0.05 & 0.2


# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

print(y_pred[:5])
print(y_test[:5])

# Calculate accuracy metric
accuracy = accuracy_score(y_pred, y_test)
mprecision = precision_score(y_test, y_pred, average='micro')
sprecision = precision_score(y_test, y_pred, average='samples')
f=f1_score(y_test, y_pred, average='micro')


#Print all values
print('The accuracy is: ', accuracy*100, '%')
print('The Micro Precision is: ', mprecision*100, '%')
print('The Sample Precision is: ', sprecision*100, '%')
print('f1:                      ', f*100 , "%")

