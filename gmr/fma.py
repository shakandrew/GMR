import os
import csv
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pathlib
import warnings
warnings.filterwarnings('ignore')

"""
*
All audio files structure is expected to be

audio ----000---000001.mp3
       |      |-000002.mp3
       |      ...
       |--001---001001.mp3
       |      |-001002.mp3
       |      ...
       |--002---002001.mp3
       |      |-002002.mp3
       |      ...
       ...
"""
from socket import gethostname

DICT_GENRES = {'Electronic': 1, 'Experimental': 2, 'Folk': 3, 'Hip-Hop': 4,
               'Instrumental': 5, 'International': 6, 'Pop': 7, 'Rock': 8}

if gethostname() == "ashumak-pc":
    AUDIO_DIR = "/home/ashumak/repo/GMR-test/fma_small/"
else:
    AUDIO_DIR = "./audio/"
N_TTF = 2048
HOP_LENGTH = 1024
FREQ_SLICES = 640
MIN_TRACK_LENGTH = 5
# DATASET_TYPE can operate with "spectrogram", "feature" values
DATASET_TYPE = "spectrogram"

def get_track_path(audio_dir, track_id):
    """
    Get audio file path with id = track_id
    :param audio_dir: str ; Path to root audio directory
    :param track_id: int ; Track ID
    :return: str ; Track path
    """
    track_id_str = f"{track_id:06d}"
    path = os.path.join(audio_dir, track_id_str[:3], track_id_str + ".mp3")
    if not os.path.exists(path):
        raise Exception(f"Cannot find file with name {track_id_str}.mp3 FMA audio audio-set*.")
    return path


def get_genres(df):
    """

    :param df:
    :return:
    """
    return {val: ind for ind, val in enumerate(df.track.genre_top.unique().tolist())}


def create_spectrogram(track_id, db=False):
    """

    :param track_id:
    :return:
    """
    filename = get_track_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    if len(y) / sr < MIN_TRACK_LENGTH:
        raise Exception("Cannot progress track due to its size. Too small.")
    spectrogram = librosa.feature.melspectrogram(y, sr, n_fft=N_TTF, hop_length=HOP_LENGTH)
    if db:
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram.T


def load_raw_dataset(filename, ds_size="small", keep_cols=[("set", "split"), ("set", "subset"), ("track", "genre_top")]):
    tracks = pd.read_csv(filename, index_col=0, header=[0, 1])
    keep_cols = [("set", "split"), ("set", "subset"), ("track", "genre_top")]
    df = tracks[keep_cols]

    if ds_size not in ("small", "medium", "large"):
        raise Exception(f"Unsupported dataset size {ds_size}.")
    df = df[df[("set", "subset")] == ds_size]

    df["track_id"] = df.index
    return df


def create_datasets(filename, genres=None, ds_size="small", ds_type="spectrogram"):
    """

    :param filename:
    :param genres:
    :param ds_size:
    :param ds_type:
    :return:
    """
    df = load_raw_dataset(filename, ds_size)

    if not genres:
        genres = get_genres(df)
        
    train_df = df[df[('set', 'split')]=="training"]
    _create_set(train_df, os.path.dirname(filename), genres, "train")
    
    valid_df = df[df[('set', 'split')]=="validation"]
    _create_set(df, os.path.dirname(filename), genres, "valid")
    
    test_df = df[df[('set', 'split')]=="test"]
    _create_set(df, os.path.dirname(filename), genres, "test")
    

################################################################################
def create_all_datasets_parallel(filename, dest_dir):
    _create_dataset_parallel(filename, dest_dir, "training", 16)
    _create_dataset_parallel(filename, dest_dir, "validation", 2)
    _create_dataset_parallel(filename, dest_dir, "test", 2)    
    
    
def _create_dataset_parallel(filename, dest_dir, set_name, threads_num=16):
    import os
    main_cmd = ""
    mult = 400
    cmd = 'python -c "import gmr.fma as fma; fma.cmd_call_create_set(\'{}\', \'{}\', \'{}\', {}, {}, {})" &'
    for i in range(threads_num):
        main_cmd += cmd.format(filename, dest_dir, set_name, i*mult, (i+1)*mult, i) 
    main_cmd += "& echo Splitting is ready!"
    print(main_cmd)
    
    if set_name !="training": 
        if os.system(main_cmd):
            raise RuntimeError("Program failed: \n {}".format(main_cmd))

    x, y = [], []
    for i in range(threads_num):
        file = np.load(os.path.join(dest_dir, f"{set_name}{i}.npz"))
        x.append(file['arr_0'])
        y.append(file['arr_1'])
    X = np.concatenate(x, axis=0)
    Y = np.concatenate(y, axis=0)
    np.savez(set_name, X, Y)
    
        
def cmd_call_create_set(
    filename, dest_dir, set_name, from_id, to_id, thread_id, ds_size="small", ds_type="spectrogram"
):
    df = load_raw_dataset(filename, ds_size)
    df = df[df[('set', 'split')]==set_name][from_id: to_id]
    _create_set(df, dest_dir, DICT_GENRES, f"{set_name}{thread_id}")
################################################################################        
        
    
def _create_set(df, dest_dir, genres, output_name):
    if DATASET_TYPE == "spectrogram":
        x, y = _create_spectrogram_dataset(df, genres)
    elif DATASET_TYPE == "feature":
        x, y = np.array([]), np.array([])
        pass
    else:
        raise Exception(f"Unsupported dataset type {DATASET_TYPE}")
    
    np.savez(os.path.join(dest_dir, output_name), x, y)


def _create_spectrogram_dataset(df, genres):
    """

    :param genres:
    :param df:
    :return:
    """
    x, y = [], []
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            spectrogram = create_spectrogram(int(row['track_id']))
            x.append(spectrogram[:FREQ_SLICES, :])
            y.append(genres[str(row[('track', 'genre_top')])])
        except Exception as e:
            print(e)
    return np.array(x), np.array(y)

def _create_feature_dataset(df, genres):
    """

    :param df:
    :param genres:
    :return:
    """
    pass