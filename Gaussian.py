# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:15:43 2022

@author: Ismaeel
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:39:52 2022

@author: Ismaeel
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from sklearn.preprocessing import MultiLabelBinarizer


df = pd.read_csv(r'C:\Users\wakob\Downloads\UniDownloads\year3\dataset1.csv',index_col=0)
df.drop(['track_id', 'album_name'], axis = 1, inplace = True)

df["artists"] = df["artists"].str.split(';')

df['explicit'] = df['explicit'].astype(bool)

track_genre_dict = {genre: i for i, genre in enumerate(df['track_genre'].unique())}
df['track_genre'] = df['track_genre'].apply(lambda x: track_genre_dict[x])


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

#all_artists = df2["artists"].apply(pd.Series).stack().unique()
#artist_dict = {artist: i for i, artist in enumerate(all_artists)}
#df2['artists'] = df2['artists'].apply(lambda x: [artist_dict[a] for a in x])

#df2 = df2.dropna()



df2 = df2.sample(20000)
all_artists = df2["artists"].apply(pd.Series).stack().unique()
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df2['artists'] = df2['artists'].apply(lambda x: [artist_dict[a] for a in x])

df2 = df2.dropna()



mlb = MultiLabelBinarizer()
df3 = df2
artists_bin = mlb.fit_transform(df3['artists'])



track_genre_bin = mlb.fit_transform(df3['track_genre'])

result = []
for x in df3.columns:
    if (x != 'track_genre' and x!='artists'):
        result.append(x)

X = df3[result].values
y = track_genre_bin


X = np.concatenate((X, artists_bin), axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#gnb_model = GaussianNB()

gnb = MultiOutputClassifier(GaussianNB())
# Fit the model to the training data
gnb.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = gnb.predict(X_test)


accuracy = accuracy_score(y_pred, y_test)
pmi = precision_score(y_test, y_pred, average='micro')
ps = precision_score(y_test, y_pred, average='samples')

print ("Gaussian")
print('The accuracy is: ',accuracy*100,'%')
print('The micro is: ',pmi*100,'%')
print('The samples is: ',ps*100,'%')