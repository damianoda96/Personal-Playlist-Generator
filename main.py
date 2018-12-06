## Deven Damiano and Nick Horvath - AI and Heuristic Programming

import sys
import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.tree import _tree
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

cid = "5ebe28f6cfa44d639ac4b91ef42444bb"
secret = "adf23eab627f477792d6bc6ca81c8fcd"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

sp.trace = False

user_id = input("Spotify ID: ")
user_playlist_id = input("Playlist ID: ")

#userid = "p0q8ccwzh5xvxzgq2jtd41abr"
#user_playlist = "57aP0BOe978aT68FeJPFPv"

playlist = sp.user_playlist(user_id, user_playlist_id)

#profile_data = sp.current_user()

songs = playlist["tracks"]["items"]
tracks = playlist["tracks"]

ids = []
artists = []
song_titles = []

#for i in range(len(songs)):
#ids.append(songs[i]["track"]["id"])

while tracks['next']:
    
    tracks = sp.next(tracks)
    
    for item in tracks["items"]:
        
        if(item['track']['id'] is not None):
            
            ids.append(item["track"]["id"])
            
            thing = item["track"]["artists"]
            
            artists.append(thing[0]["name"])
            
            song_titles.append(item["track"]["name"])

#print(thing[0]["name"], "- ", item["track"]["name"])

#print(item["track"]["id"])
#print(item["track"]["artists"])

features = []

for i in range(0, len(ids), 50):
    
    audio_features = sp.audio_features(ids[i:i+50])
    
    for track in audio_features:
        
        features.append(track)

df = pd.DataFrame(features)

#df.drop(['analysis_url', 'id', 'track_href', 'type', 'uri'], axis=1)

df1 = pd.DataFrame(artists, columns=['artist'])

df2 = pd.DataFrame(song_titles, columns=['song_title'])

targets = []

for i in range(len(artists)):
    
    targets.append(1)


df3 = pd.DataFrame(targets, columns=['target'])

if(user_id == "p0q8ccwzh5xvxzgq2jtd41abr")
{
    test_df = pd.read_csv('data/deven.csv')
}
else
{
    test_df = pd.read_csv('data/nick.csv')
}

train_data, test_data = train_test_split(test_df, test_size = .60)

frames = [df, df1, df2, df3]

df4 = pd.concat(frames, axis=1)

frames2 = [df4, train_data]

df5 = pd.concat(frames2, axis=0, sort=False)

df5.to_csv('data/user.csv', sep=',', encoding='utf-8')

attribute_list = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']

train, test_data_2 = train_test_split(df5, test_size = .15)

test_list = [test_data, test_data_2]

all_test_data = pd.concat(test_list, axis=0, sort=False)

all_test_data = all_test_data.reset_index(drop=True)

c = tree.DecisionTreeClassifier(min_samples_split=10)

X_train = train[attribute_list]
y_train = train['target']

#X_test = test[attribute_list]
#y_test = test['target']

#Take the first 500 songs of the original dataset and add to training csv
#so we have an idea of songs we do not like

dt = c.fit(X_train, y_train)

#write tree to file

out_file_joblib_name = (user_id + ".joblib")
    
joblib.dump(dt, "models/" + out_file_joblib_name)

out_file_dot_name = (user_id + ".dot")

out_file_png_name = (user_id + ".png")

tree.export_graphviz(dt, out_file=("visuals/" + out_file_dot_name), class_names = True, feature_names = attribute_list)
    
#print(classifier_list)
    
from subprocess import call
call(['dot', '-Tpng', ("visuals/" + out_file_dot_name), '-o', ("visuals/" + out_file_png_name), '-Gdpi=600'])

X_test = all_test_data[attribute_list]
y_test = all_test_data['target']

y_pred = dt.predict(X_test)

counter = 0

print("")

print("SUGGESTED PLAYLIST FOR USER: ", user_id)

print("----------------------")

for i in range(len(y_pred)):  #for all results
    
    if(y_pred[i] == 1):
        
        row = all_test_data.loc[i]
        
        print(row['artist'], '-', row['song_title'])
        
        counter = counter + 1

    if(counter == 15):
        
        break

print("----------------------")
print("")

#print(all_test_data)

#get test data artist and song title at that index and print it!!!!

#print(y_pred)

##feed a random selections of songs to the tree, if 1, add 10 songs to a playist, then output the playlist
##should be unique for each user we feed the program

#for i in range(len(y_pred)):

#if(y_pred[i] == 1):

#find connected song title and artist


#score = accuracy_score(y_test, y_pred) * 100

#print("Accuracy: ", score)
