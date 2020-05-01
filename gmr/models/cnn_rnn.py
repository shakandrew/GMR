import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

import librosa
import librosa.display
import matplotlib.pyplot as plt

dict_genres = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
               'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}


def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    cnn = Conv2D(filters=16, kernel_size=(3, 1), strides=1, padding='valid', activation='relu')(input_layer)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding='valid', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 1), strides=1, padding='valid', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 1), strides=1, padding='valid', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 1), strides=1, padding='valid', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Flatten()(cnn)
    rnn = MaxPooling2D(pool_size=(4, 2))(input_layer)

    rnn = Lambda(lambda x: K.squeeze(x, axis=-1))(rnn)

    rnn = Bidirectional(GRU(64))(rnn)  # default merge mode is concat

    merged = concatenate([cnn, rnn], axis=-1)

    model_output = Dense(8, activation='softmax')(merged)
    model = Model(input_layer, model_output)

    optimizer = RMSprop(lr=0.0005)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model
