import pandas as pd
import numpy as np
import io
import os
import base64
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from math import pi


def spider_plot(df_with_titles, fig):
    
    categoricals = ['track_id', 'track_name', 'artist_name']
    misleading = ['key', 'time_signature', 'popularity', 'mode', 'tempo', 'duration_ms']

    unwanted = categoricals + misleading
    
    # number of variables
    df = df_with_titles.copy()
    for col in unwanted:
        if col in df.columns:
            df = df.drop(unwanted, axis=1)
    categories = df.columns
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    angles = np.array(angles)
    
    # # Initialize the spider plot
    # # set every background to be transparent
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111,
                         polar=True)
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    
    # # Get the audio features of the inputed song and repeat the first value at the end
    # # We need to repeat the first value in each row of the dataframe to close the circular graph:
    song_values = df.iloc[0].values.flatten().tolist()
    song_values += song_values[:1]
    # convert back to numpy array because we're going to be doing math later
    song_values = abs(np.array(song_values))
    
    # # plot the Base song
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

    # # for use in setting the maximum y limit
    maximum_diff = 0
    
    # # set number of nearest neighbors that will appear on the graph
    num_neighbors = 3
    # # "3" is currently how many of the top 9 closest songs we are choosing to show
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
        
    # # Draw one axis per variable, add x labels
    ax.set_xticks(angles[:-1], )
    ax.set_thetagrids(angles[:-1] * 180/pi, labels=categories, fontsize=14,)
    ax.tick_params(axis='x', color='gray', labelcolor='gray', labelsize=14)
    
    # # Draw ylabels    
    # # set theta position to 0
    ax.set_rlabel_position(35.5)
    # # make the tick lengths (and label names since the lengths are the labels)
    yticks = [round(0.2 * maximum_diff, 2), 
              round(0.4 * maximum_diff, 2), 
              round(0.6 * maximum_diff, 2), 
              round(0.8 * maximum_diff, 2), 
              round(1.0 * maximum_diff, 2)]
    ax.set_yticks(yticks, )
    ax.tick_params(axis='y', color='gray', labelcolor='gray', labelsize=12)
    
    ax.spines['polar'].set_visible('False')
    # set maximum y limit to the largest prong of our web, 
    # that way the plot is exactly as big as it need to be, 
    # and no larger
    ax.set_ylim(0, 1.1 * maximum_diff)
    
    fig.suptitle(f'Audio Features of your song (in green) and our Recommendations for you', color='white')
    
    # show the plot
    legend = ax.legend(facecolor='gray', framealpha=0.15)
    for text in legend.get_texts():
        text.set_color('white')


def preprocess(df):
    """ normalizes pandas df.
    Removes unecessary columns """
    drop_cols = ['duration_ms', 'key', 'mode', 'time_signature', 'popularity', 'tempo']
    drop_cols += ['track_id', 'track_name', 'artist_name']
    for col in drop_cols:
        if col in list(df.columns):
            df = df.drop(columns=col)
    return df

def create_model(X, n_neighbors=10):
    """ Insantiate nearest neighbor model """
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    model.fit(X)
    return model

def suggest_songs(source_song, songs_df, y, model):
    """ Preprecesses source song, use it to make suggestions from the database """
    source_song = preprocess(source_song)
    recommendations = model.kneighbors(source_song)[1][0]
    # normalize dataset, our graph likes normalized data
    numeric_cols = songs_df.select_dtypes(include=np.number).columns
    df_num = songs_df.select_dtypes(include=np.number)
    songs_df_norm = songs_df.copy()
    songs_df_norm[numeric_cols] = (df_num - df_num.mean()) / df_num.std()
    return songs_df_norm.iloc[recommendations]
