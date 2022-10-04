import pickle
import os
from os.path import join as p_join
import pickle
from statistics import mode

from lungai.model import sequential_model

def load_trained_model(path: str):
    shape_path = p_join(path, "data_shape.tuple")
    with open(shape_path, "rb") as file:
        input_shape, n_classes = pickle.load(file)

    model = sequential_model(input_shape, n_classes)
    model.load_weights(p_join(path, "w"))
    return model
