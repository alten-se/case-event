from typing import Tuple

from keras.models import Sequential
from keras.layers import Dense, Dropout # "CudNNLSTM" is GPU optimized variant of "LTSM"
import tensorflow as tf


print(tf.config.list_physical_devices())

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("CUDA enbabled GPU detected!")
    from keras.layers import CuDNNLSTM as LSTM
else:
    print("no cuda enbaled gpu, falling back to CPU")
    from keras.layers import LSTM

def rnn_model(shape: Tuple[int, int]) -> Sequential:
    """_summary_

    Args:
        shape (time_steps: int,  features: int): the shape of the input that ai should work with

    Return:
        rnn_model: tensor model
    """
    model = Sequential()
    time_steps, features = shape

    model.add(LSTM(
    128,
    input_shape=(time_steps, features),
    return_sequences=True))
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

    n_disease_classes = 7
    model.add(Dense(n_disease_classes, activation="softmax"))
    
    return model