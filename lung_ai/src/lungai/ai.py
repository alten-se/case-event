from __future__ import annotations

from typing import Tuple, List, Any
from warnings import catch_warnings
from numpy import ndarray

import os, pickle

import numpy as np
from soundfile import SoundFile

from lungai.model import rnn_model
from lungai.data_extraction import extract_mfccs
from lungai.data_gen import DataGenerator
from lungai.train import train


class AI:
    """
    High level handler of rnn model
    """

    def __init__(self, input_shape, n_classes) -> None:
        self.model = rnn_model(input_shape, n_classes)
        self.io_shape = (input_shape, n_classes)
        self.label_dict = None

    @staticmethod
    def get_paths(path: str):
        w_path = os.path.join(path, "w.h5")
        shape_path = os.path.join(path, "data_shape.tuple")
        label_dict_path = os.path.join(path, "label.dict")
        return w_path, shape_path, label_dict_path

    def save(self, path: str) -> None:
        """saves weights and some meta about model

        Args:
            path (str): path/to/save/dir
        """
        w_path, shape_path, label_dict_path = self.get_paths(path)

        self.model.save_weights(w_path)
        with open(shape_path, mode="wb+") as file:
            pickle.dump(self.io_shape, file)

        if self.label_dict:
            with open(label_dict_path, mode="wb+") as file:
                pickle.dump(self.label_dict, file)

    @classmethod
    def load(cls, path: str) -> AI:
        w_path, shape_path, label_dict_path = cls.get_paths(path)

        assert (os.path.exists(w_path))
        assert (os.path.exists(shape_path))
        with open(shape_path, mode="rb") as file:
            shape_io = pickle.load(file)
        ai = cls(*shape_io)
        ai.model.load_weights(w_path)

        if os.path.exists(label_dict_path):
            with open(label_dict_path, mode="rb") as file:
                ai.label_dict = pickle.load(file)

        return ai

    def predict_one(self, input: ndarray) -> Tuple[str, float]:
        """_summary_

        Args:
            input (ndarray): mfccs from single records. shape = (time_steps, features)

        Returns: 
            Tuple(
                str: predicted label,
                float: confidence level between 0 to 1.0
            )
        """

        pred = np.array(self.model(input[np.newaxis, :]))
        index = np.argmax(pred)

        confidence = pred[0, index]

        if self.label_dict is None:
            return index, confidence

        inv_map = invert_dict(self.label_dict)

        return inv_map[index], confidence

    def predict_sound(self, file: str):
        s_file = SoundFile(file, "r")
        mffcs = extract_mfccs(s_file)
        return self.predict_one(mffcs.T)

    def train(self, train_set, validate_set = None, epochs = 3, **kwargs):   
        train_gen = DataGenerator(train_set, **kwargs)
        if validate_set:
            valid_gen = DataGenerator(validate_set, **kwargs)
        else:
            valid_gen = None
        self.model = train(train_gen, valid_gen, self.model, epochs)

def invert_dict(k_v: dict) -> dict:
    """Creates a new dict that is the inverted input dict.

    Args:
        k_v (dict): a dict{ keys, values}

    Returns:
        dict: v_k
    """
    return {v: k for k, v in k_v.items()}
