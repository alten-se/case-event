import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from os.path import join as p_join

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from lungai.load_model import load_trained_model
from lungai.data_extraction import get_data
from lungai.paths import TRAINED_MODELS_PATH, DATA_PATH
from lungai.evaluate import eval, eval_sound
from lungai.clear import clear

from lungai.model import AI


def get_model():
    ai = AI.load(p_join(TRAINED_MODELS_PATH, "dummy"))
    return ai.model

def pretty_print(label: str, confidence: float):
    print("Prediction: {label}, prob. = {confidence:.3%}".format(
            label=label, 
            confidence=confidence))

def test_load():
    data, labels, label_dict = get_data()
    model = get_model()

    test_inds = [0, 1]

    predictions = [ eval(data_point, model, label_dict) for data_point in data[test_inds]]
    
    for label, confidence in predictions:
        pretty_print(label, confidence)
        
    

def test_eval_sound():
    _, _, label_dict = get_data()
    model = get_model()
    record_name = "101_1b1_Al_sc_Meditron.wav"
    label, conf = eval_sound(
        p_join(DATA_PATH, record_name),
        model,
        label_dict
        )
    pretty_print(label, conf)




if __name__ == "__main__":
    test_eval_sound()
