import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

##Data Management
##Takes in raw data source downloaded from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
df = pd.read_csv('Spotify-Dupes.csv',index_col=0)
##Drops columns that we don't need for analysis of individual songs
df.drop(['track_id', 'album_name'], axis = 1, inplace = True)

# Use the `str.split` method to split the `artists` column into a list of artists
df["artists"] = df["artists"].str.split(';')

# Convert the `explicit` column to a boolean
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


#Drops rows that are of the selected genre
#Done to remove some artists as the number of artists caused the data to take too long to load due to the Multi-Label Binarizer
#We selected these genres as we believed they were some of the least popular genre and with the less re-occuring artists
df = df[~df['track_genre'].apply(lambda x: 'j-pop' in x or 'anime' in x or 'black-metal' in x or 'bluegrass' in x or 'brazil' in x or 'cantopop' in x
                                  or 'french' in x or 'german' in x or 'indian' in x or 'iranian' in x or 'j-dance' in x or 'j-idol' in x
                                  or 'j-rock' in x or'k-pop' in x or 'malay' in x or 'mandopop' in x or 'swedish' in x or 'turkish' in x or 'world-music' in x)]

df = df.sample(200)
#Retrieves all the unique genres present in the dataframe
all_genres = df["track_genre"].apply(pd.Series).stack().unique()
#Enumerates all the genres and stores them to a dictionary
track_genre_dict = {genre: i for i, genre in enumerate(all_genres)}
df['track_genre'] = df['track_genre'].apply(lambda x: [track_genre_dict[a] for a in x])
#Removes any duplicate genre tags that may have occured due to a song being featured on multiple albums
df['track_genre'] = df['track_genre'].apply(set)
df['track_genre'] = df['track_genre'].apply(list)

#Retrieves all the unique artists present in the dataframe
all_artists = df["artists"].apply(pd.Series).stack().unique()
#Enumerates all the artists and stores them to a dictionary
artist_dict = {artist: i for i, artist in enumerate(all_artists)}
df['artists'] = df['artists'].apply(lambda x: [artist_dict[a] for a in x])
#Removes any duplicate artists tags
df['artists'] = df['artists'].apply(set)
df['artists'] = df['artists'].apply(list)

#Drops all blank rows due to grouping
df = df.dropna()
#Initializes the multi-label binerizer, used for the multi-label classification
mlb = MultiLabelBinarizer()
##Fits the artists and trackgenres to the binerizer and outputs the matrix
artists_bin = mlb.fit_transform(df['artists'])
track_genre_bin = mlb.fit_transform(df['track_genre'])

##Takes in all columns that are needed for the X vaslues
result = []
for x in df.columns:
    if (x != 'track_genre' and x!='artists'):
        result.append(x)

X = df[result].values
# Concatenate the binerized artist columns to the input data
X = np.concatenate((X, artists_bin), axis=1)
#Sets the binerized track genres as the y values
y = track_genre_bin


##Random Forest Model
#Initialize the MultiOutputClassifier to the random forest model
#Set to run on 4 cpu cores with 5 estimators
clf = MultiOutputClassifier(RandomForestClassifier(n_jobs = 4, n_estimators = 5))

#Splits the data into training and testing data
#Sets the test size to 5% as wanted to give it as much learning oppurtunity and 5% is still 2853 rows of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#Fit the model to the training data
clf.fit(X_train, y_train)

#Make predictions on the testing data
y_pred = clf.predict(X_test)

#Calculate metrics
accuracy = accuracy_score(y_pred, y_test)
mprecision = precision_score(y_test, y_pred, average='micro', zero_division = 0)
macprecision = precision_score(y_test, y_pred, average='macro', zero_division = 0)
wprecision = precision_score(y_test, y_pred, average='weighted', zero_division = 0)
sprecision = precision_score(y_test, y_pred, average='samples', zero_division = 0)
f=f1_score(y_test, y_pred, average='micro')

#Output results for Random Forest
print("Random Forest Results:")
print('The accuracy is: ',accuracy*100,'%')
print('The Micro Precision is: ',mprecision*100,'%')
print('The Macro Precision is: ',macprecision*100,'%')
print('The Weight Precision is: ',wprecision*100,'%')
print('The Sample Precision is: ',sprecision*100,'%')
print("f1 =  ", f*100 , "%")

##NN
#Initialize the MultiOutputClassifier to the MLP Neural Network Model
clf = MultiOutputClassifier(MLPClassifier(random_state=1))

#Splits the data into training and testing data
#Sets the test size to 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Change the test size 0.05 & 0.2

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy metrics
accuracy = accuracy_score(y_pred, y_test)
mprecision = precision_score(y_test, y_pred, average='micro', zero_division = 0)
sprecision = precision_score(y_test, y_pred, average='samples', zero_division = 0)
f=f1_score(y_test, y_pred, average='micro')


#Print all metrics
print("Neural Network Results:")
print('The accuracy is: ', accuracy*100, '%')
print('The Micro Precision is: ', mprecision*100, '%')
print('The Sample Precision is: ', sprecision*100, '%')
print('f1:', f*100 , "%")

##Gaussian
#Initialize the MultiOutputClassifier to the GaussinNB Model
gnb = MultiOutputClassifier(GaussianNB())
#Splits the data into training and testing data
#Sets the test size to 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#Fit the model to the training data
gnb.fit(X_train, y_train)
#Make predictions on the testing data
y_pred = gnb.predict(X_test)

# Calculate accuracy metrics
accuracy = accuracy_score(y_pred, y_test)
pmi = precision_score(y_test, y_pred, average='micro', zero_division = 0)
pma = precision_score(y_test, y_pred, average='macro', zero_division = 0)
pw = precision_score(y_test, y_pred, average='weighted', zero_division = 0)
ps = precision_score(y_test, y_pred, average='samples', zero_division = 0)
f=f1_score(y_test, y_pred, average='micro')

#Print all metrics
print("Gaussian Results:")
print('The accuracy is: ',accuracy*100,'%')
print('The micro is: ',pmi*100,'%')
print('The macro is: ',pma*100,'%')
print('The weighted is: ',pw*100,'%')
print('The samples is: ',ps*100,'%')
print("f1 =  ", f*100 , "%")

##Decision
#Initialize the MultiOutputClassifier to the Decision Tree Model
dt = MultiOutputClassifier(DecisionTreeClassifier())
#Splits the data into training and testing data
#Sets the test size to 5%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
#Fit the model to the training data
dt.fit(X_train,y_train)
# Make predictions on the testing data
y_pred = dt.predict(X_test)

# Calculate accuracy metric
accuracy = accuracy_score(y_pred, y_test)
mprecision = precision_score(y_test, y_pred, average='micro', zero_division = 0)
macprecision = precision_score(y_test, y_pred, average='macro', zero_division = 0)
wprecision = precision_score(y_test, y_pred, average='weighted', zero_division = 0)
sprecision = precision_score(y_test, y_pred, average='samples', zero_division = 0)
f=f1_score(y_test, y_pred, average='micro')

#Print all metrics
print("Decision Tree Results")
print('The accuracy is: ',accuracy*100,'%')
print('The Micro Precision is: ',mprecision*100,'%')
print('The Macro Precision is: ',macprecision*100,'%')
print('The Weight Precision is: ',wprecision*100,'%')
print('The Sample Precision is: ',sprecision*100,'%')
print("f1 =  ", f*100 , "%")

##SVM
# Initialize the SVC classifier
clf = SVC()
#Splits the data into training and testing data
#Sets the test size to 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42) 
#Initialize the MultiOutputClassifier with the SVC classifier
multi_clf = MultiOutputClassifier(clf)

# Fit the classifier to the input data and labels
multi_clf.fit(X, y)

# Use the trained classifier to make predictions on the test data
predictions = multi_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(predictions,y_test)
precision = precision_score(y_test, predictions, average='micro', zero_division = 0)
ps = precision_score(y_test, predictions, average='samples', zero_division = 0)
f=f1_score(y_test, predictions, average='micro')

#Print all metrics
print("SVM Result:")
print("Accuracy:", accuracy)
print("Micro Precision:", precision)
print("Sample Precision:", ps)
print("F1 Score:",f)