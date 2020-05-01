import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from gmr.models.cnn_rnn import build_model
import matplotlib.pyplot as plt


def train(x_train, y_train, x_valid, y_valid, frames_num=640, frequency_num=128, callbacks=None, batch_size=64,
          epochs=100):
    input_shape = (frames_num, frequency_num, 1)
    model = build_model(input_shape)
    print("Start training...")
    checkpoint_callback = ModelCheckpoint('./models/parallel/weights.best.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    reducelr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01, verbose=1)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid),
                        verbose=1, callbacks=[checkpoint_callback, reducelr_callback])
    return model, history


def show_summary_stats(history):
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    x_train, y_train = np.load("train.npz")
    x_valid, y_valid = np.load("valid.npz")

    mdl, hist = train(x_train, y_train, x_valid, y_valid)
