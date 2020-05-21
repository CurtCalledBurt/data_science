import pandas as pd
import numpy as np
import os
import io
from flask import Flask, Response
from flask_cors import CORS
from sqlalchemy import create_engine
from functions import spider_plot, preprocess, create_model, suggest_songs

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
CORS(app)

# Couple of ways included to create the pandas dataframe, one from sqlite, one via path, and finally one via github

# Sqlite option
# # songs_df =  pd.read_sql_table('songs', 'sqlite:///db.sqlite3')

# path option
# songs_df = pd.read_csv('../Data/SpotifyAudioFeaturesApril2019_duplicates_removed.csv')

# github option
infile = "https://raw.githubusercontent.com/spotify-recommendation-engine-3/data_science/master/Data/SpotifyAudioFeaturesApril2019_duplicates_removed.csv"
songs_df = pd.read_csv(infile)

y = songs_df[songs_df.columns[:3]]
X = songs_df[songs_df.columns[3:]]

my_model = create_model(preprocess(X))

@app.route('/last_hope', methods=['GET', 'POST'])
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    song_df = songs_df.sample()
    song_df = song_df.iloc[:, 3:]
    songs_to_plot = suggest_songs(song_df, songs_df, y, my_model)
    fig = Figure(figsize=(9, 9),
                edgecolor='gray')
    spider_plot(songs_to_plot, fig)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8000)
