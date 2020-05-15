import pandas as pd
import numpy as np
import io
from math import pi
import base64
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS
from sqlalchemy import create_engine


app = Flask(__name__)
CORS(app)

engine = create_engine('sqlite:///db.sqlite3', echo=False)
songs_df =  pd.read_sql_table('songs', 'sqlite:///db.sqlite3')
# songs_df = pd.read_csv('/Users/curtismckendrick/lambda school repositories/build_weeks/Spotify_Model/data_science/Data/SpotifyAudioFeaturesApril2019_duplicates_removed.csv')

# infile = "https://raw.githubusercontent.com/spotify-recommendation-engine-3/data_science/master/Data/SpotifyAudioFeaturesApril2019_duplicates_removed.csv"
# songs_df = pd.read_csv(infile)
y = songs_df[songs_df.columns[:3]]
X = songs_df[songs_df.columns[3:]]

def spider_plot(df_with_titles):
    
    categoricals = ['track_id', 'track_name', 'artist_name']
    misleading = ['key', 'time_signature', 'popularity', 'mode', 'tempo', 'duration_ms']

    unwanted = categoricals + misleading
    
    # number of variables
    df = df_with_titles.copy()
    df = df.drop(unwanted, 
                 axis=1)
    categories = df.columns
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    angles = np.array(angles)
    
    # Initialize the spider plot
    # set every background to be transparent
    fig = plt.figure(figsize=(9, 9), 
                     edgecolor='gray')
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, 
                         polar=True)
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    
    # Get the audio features of the inputed song and repeat the first value at the end
    # We need to repeat the first value in each row of the dataframe to close the circular graph:
    song_values = df.iloc[0].values.flatten().tolist()
    song_values += song_values[:1]
    # convert back to numpy array because we're going to be doing math later
    song_values = abs(np.array(song_values))
    
    
    # plot the Base song
    ax.plot(angles,
           song_values,
           linewidth=3,
           linestyle='solid',
           label=df_with_titles.iloc[0]['track_name'],
           color='limegreen',
           alpha=1)
    # fill the base song
    ax.fill(angles,
           song_values,
           color='lime',
           alpha=0.33)
    

    # for use in setting the maximum y limit
    maximum_diff = 0
    
    # set number of nearest neighbors that will appear on the graph
    num_neighbors = 3
    # "3" is currently how many of the top 9 closest songs we are choosing to show
    for i in range(num_neighbors):
        
        # Again repeat the first value in the array to close the circle
        # skipping the first row, because that's the target song
        diff_values = df.iloc[i+1].values.flatten().tolist()
        diff_values += diff_values[:1]
        diff_values = abs(np.array(diff_values))
        
        colors=['b', 'r', 'orange', 'y', 'k', 'm', 'c', 'w', 'pink']
        # plot the recommendations
        ax.plot(angles, 
                diff_values, 
                linewidth=2, 
                linestyle='solid', 
                label=df_with_titles.iloc[i+1]['track_name'],
               color=colors[i])
        # fill the recommendations
        ax.fill(angles, 
                diff_values,
                color=colors[i],
                alpha=0)
        
        # check for new maximum y limit
        if max(diff_values) > maximum_diff:
            maximum_diff = max(diff_values)
        
    # Draw one axis per variable, add x labels
    plt.xticks(angles[:-1], 
               categories, 
               color='gray', 
               size=14)
    
    # Draw ylabels    
    # set theta position to 0
    ax.set_rlabel_position(22.5)
    # make the tick lengths (and label names since the lengths are the labels)
    yticks = [round(0.2 * maximum_diff, 2), 
              round(0.4 * maximum_diff, 2), 
              round(0.6 * maximum_diff, 2), 
              round(0.8 * maximum_diff, 2), 
              round(1.0 * maximum_diff, 2)]
    plt.yticks(yticks, 
               yticks, 
               color = 'gray', 
               fontsize=12)
    
    ax.spines['polar'].set_visible('False')
    # set maximum y limit to the largest prong of our web, 
    # that way the plot is exactly as big as it need to be, 
    # and no larger
    plt.ylim(0, 1.1 * maximum_diff)
    
    plt.title(f'Audio Features of your song (in bold) and our Recommendations for you', 
              color='grey')
    
    # show the plot
    ax.legend()

    # # DEPRECATED: return encoded figure
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return pic_hash

    # return fig

def preprocess(df):
    """ normalizes pandas df.
    Removes unecessary columns """
    drop_cols = ['duration_ms', 'key', 'mode', 'time_signature', 'popularity', 'tempo']
    drop_cols += ['track_id', 'track_name', 'artist_name']
    for col in drop_cols:
        if col in list(df.columns):
            df = df.drop(columns=col)
    # drop_cols = ['key', 'mode', 'time_signature', 'popularity', 'tempo']
    # artist_name,track_id,track_name,acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,time_signature,valence,popularity
    # df = df.drop(columns=drop_cols)
    return df

def create_model(X, n_neighbors=10):
    """ Insantiate nearest neighbor model """
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    model.fit(X)
    return model

def suggest_songs(source_song, model):
    """ Preprecesses source song, use it to make suggestions from the database """
    source_song = preprocess(source_song)
    # source_song = scaler.fit_transform(source_song)
    recommendations = model.kneighbors(source_song)[1][0]
    # normalize dataset, our graph likes normalized data
    numeric_cols = songs_df.select_dtypes(include=np.number).columns
    df_num = songs_df.select_dtypes(include=np.number)
    songs_df_norm = songs_df.copy()
    songs_df_norm[numeric_cols] = (df_num - df_num.mean()) / df_num.std()
    pic_hash = spider_plot(songs_df_norm.iloc[recommendations])
    recommendations_dict = y.iloc[recommendations][1:].reset_index(drop=True).T.to_dict()
    recommendations_dict['encoded_image'] = pic_hash
    # print(recommendations_dict)
    return recommendations_dict

my_model = create_model(preprocess(X))

@app.route('/pred', methods=['GET'])
def returnAll():
    song_dict = {"artist_name":"YG","track_id":"2RM4jf1Xa9zPgMGRDiht8O","track_name":"Big Bank feat. 2 Chainz, Big Sean, Nicki Minaj","acousticness":0.00582,"danceability":0.743,"duration_ms":238373,"energy":0.339,"instrumentalness":0.0,"key":1,"liveness":0.0812,"loudness":-7.678,"mode":1,"speechiness":0.409,"tempo":203.927,"time_signature":4,"valence":0.118,"popularity":15}
    song_dict.update((x, [y]) for x, y in song_dict.items())
    song_df = pd.DataFrame.from_dict(song_dict)
    song_df = song_df.iloc[:, 3:]
    # song_df = song_df[song_df.columns[3:]]
    
    result = suggest_songs(song_df, my_model)

    return jsonify(result)
    # return f"Hello World"

@app.route('/pred', methods=['POST'])
def runPred():
    
    input_song = request.get_json(force=True)
    input_song.update((x, [y]) for x, y in input_song.items())

    song_df = pd.DataFrame.from_dict(input_song)
    song_df = song_df.iloc[:, 3:]
    # song_df = song_df[song_df.columns[3:]]
    
    results = suggest_songs(song_df, my_model)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
