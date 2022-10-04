from typing import Tuple

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, CuDNNGRU, GRU, LeakyReLU, Add, Layer
from keras.engine.keras_tensor import KerasTensor
from keras.utils import plot_model

import tensorflow as tf

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("CUDA enbabled GPU detected!")
    DynGRU = CuDNNGRU  
else:
    print("no cuda enbaled gpu, falling back to CPU")
    DynGRU = GRU
    

def left1(input_layer: Input) -> KerasTensor:
    l1 = DynGRU(64, return_sequences=True)(input_layer)
    l2 = LeakyReLU()(l1)

    l3 = DynGRU(128, return_sequences=True)(l2)
    l4 = LeakyReLU()(l3)   

    return l4
    
def right1(input_layer: Input) -> KerasTensor:
    l1 = DynGRU(32, return_sequences=True)(input_layer)
    l2 = LeakyReLU()(l1)

    l3 = DynGRU(128, return_sequences=True)(l2)
    l4 = LeakyReLU()(l3)   

    return l4
    
def left2(input_layer: Layer) -> KerasTensor:
    l1 = DynGRU(128, return_sequences=True)(input_layer)
    l2 = LeakyReLU()(l1)

    l3 = DynGRU(32)(l2)
    l4 = LeakyReLU()(l3)   

    return l4   

def right2(input_layer: Layer) -> KerasTensor:
    l1 = DynGRU(64, return_sequences=True)(input_layer)
    l2 = LeakyReLU()(l1)

    l3 = DynGRU(32)(l2)
    l4 = LeakyReLU()(l3)   

    return l4   

def left3(input_layer: Layer) -> KerasTensor:
    l1 = Dense(64)(input_layer)
    l2 = LeakyReLU()(l1)
    l3 = Dropout(0.2)(l2)

    l4 = Dense(16)(l3)
    l5 = LeakyReLU()(l4)   

    return l5   

def right3(input_layer: Layer) -> KerasTensor:
    l1 = Dense(32)(input_layer)
    l2 = LeakyReLU()(l1)
    l3 = Dropout(0.2)(l2)

    l4 = Dense(16)(l3)
    l5 = LeakyReLU()(l4)   

    return l5   

    
def functional_model(input_shape: Tuple, n_classes: int) -> Model:

    input_layer = Input(input_shape)
    add1 = Add()([left1(input_layer), right1(input_layer)])

    l2 = left2(add1)
    r2 = right2(add1)
    add2 = Add()([l2, r2])

    l3 = left3(add2)
    r3 = right3(add2)

    add3 = Add()([l3, r3])
    stem1 = Dense(16)(add3)
    stem2 = LeakyReLU(16)(stem1)
    stem3 = Dropout(0.2)(stem2)
    stem4 = Dense(n_classes, activation="softmax")(stem3)

    model = Model(inputs=input_layer, outputs=stem4)

    return model


def sequential_model(input_shape: Tuple, n_classes: int) -> Sequential:
    """_summary_i

    Args:
        shape (time_steps: int,  features: int): the shape of the input that ai should work with

    Return:
        rnn_model: tensor model
    """
    
    model = Sequential()

    model.add(DynGRU(
        128,
        input_shape=input_shape,
        return_sequences=True
    ))

    model.add(Dropout(0.2))

    model.add(DynGRU(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(DynGRU(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(DynGRU(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(DynGRU(128, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation="softmax"))

    return model
        
if __name__ == "__main__":
    input_shape = (None, 40)
    model = functional_model(input_shape, n_classes=6)
    model.summary()
    plot_model(model, show_shapes=True)
    print(model)