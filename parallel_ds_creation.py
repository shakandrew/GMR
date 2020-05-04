import sys
import numpy as np
import pandas as pd
from gmr import fma

if __name__ == "__main__":
    filename = sys.argv[1]
    tracks = pd.read_csv(filename, index_col=0, header=[0, 1])
    keep_cols = [("set", "split"), ("set", "subset"), ("track", "genre_top")]
    df = tracks[keep_cols]
    df = df[df[("set", "subset")] == "small"]
    df["track_id"] = df.index
    genres = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
                   'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}
    fma._create_set(df, genres, sys.argv[5], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
