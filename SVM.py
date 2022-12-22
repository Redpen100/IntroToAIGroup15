#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:12:51 2022

@author: osmanmurad
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score


df = pd.read_csv('spotifydataset.csv',index_col=0)
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

df2 = df2.sample(10000)
all_artists = df2["artists"].apply(pd.Series).stack().unique()
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df2['artists'] = df2['artists'].apply(lambda x: [artist_dict[a] for a in x])

df2 = df2.dropna()



mlb = MultiLabelBinarizer()
df3 = df2
artists_bin = mlb.fit_transform(df3['artists'])
track_genre_bin = mlb.fit_transform(df3['track_genre'])
print(df3)
#print(df2)
#print(df2.columns)

#df2.columns = ["track_name","artists","popularity","duration_ms","explicit","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","track_genre"]
#for label in df2.columns:
    #df2[label] = LabelEncoder().fit(df2[label]).transform(df2[label])

# Create our X and y data    
result = []
for x in df3.columns:
    if (x != 'track_genre' and x!='artists'):
        result.append(x)

X = df3[result].values
#track_genre_bin.to_csv('TestDataExport.csv')
y = track_genre_bin

# Concatenate the transformed `artists` and `track_genre` columns to the input data
X = np.concatenate((X, artists_bin), axis=1)

# Initialize the MultiOutputClassifier
clf = MultiOutputClassifier(RandomForestClassifier(n_jobs = 4,n_estimators = 1))

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
sprecision = precision_score(y_test, y_pred, average='samples')

print('The accuracy is: ',accuracy*100,'%')
print('The Micro Precision is: ',mprecision*100,'%')
print('The Sample Precision is: ',sprecision*100,'%')


# Initialize the SVC classifier
clf = SVC()

# Initialize the MultiOutputClassifier with the SVC classifier
multi_clf = MultiOutputClassifier(clf)

# Fit the classifier to the input data and labels
multi_clf.fit(X, y)

# Use the trained classifier to make predictions on the test data
predictions = multi_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(predictions ,y_test)
precision = precision_score(y_test, predictions, average='micro')
print("Accuracy:", accuracy)
print("Precision:", precision)
f=f1_score(y_test, y_pred, average='micro')
print("F1 Score:",f)
