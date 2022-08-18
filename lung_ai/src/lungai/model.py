from typing import Tuple

from keras.models import Sequential
# "CudNNLSTM" is GPU optimized variant of "LTSM"
from keras.layers import Dense, Dropout
import tensorflow as tf



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
