from __future__ import annotations
from typing import Tuple

from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

import pickle
import os

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("CUDA enbabled GPU detected!")
    from keras.layers import CuDNNLSTM as LSTM  
else:
    print("no cuda enbaled gpu, falling back to CPU")
    from keras.layers import LSTM

def rnn_model(input_shape: Tuple, n_classes: int) -> Sequential:
    """_summary_

    Args:
        shape (time_steps: int,  features: int): the shape of the input that ai should work with

    Return:
        rnn_model: tensor model
    """
    model = Sequential()

    model.add(LSTM(
        128,
        input_shape=input_shape,
        return_sequences=True
    ))

    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation="softmax"))

    return model

class AI:
    def __init__(self, input_shape, n_classes) -> None:
        self.model = rnn_model(input_shape, n_classes)
        self.io_shape =(input_shape, n_classes)
    
    def save(self, path: str) -> None:
        w_path = os.path.join(path, "w.h5")
        self.model.save_weights(w_path)
        shape_path = os.path.join("data_shape.tuple")
        with open(shape_path, mode="wb+") as file:
            pickle.dump(self.io_shape, file)

    @classmethod
    def load(cls, path) -> AI:
        w_path = os.path.join(path, "w.h5")
        shape_path = os.path.join(path, "data_shape.tuple")
        assert(os.path.exists(w_path))
        assert(os.path.exists(shape_path))
        with open(shape_path, mode="rb") as file:
            shape_io = pickle.load(file)
        ai = cls(*shape_io)
        ai.model.load_weights(w_path)

        return ai

        
