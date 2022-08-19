
import numpy as np
from os.path import join

from lungai.tf_config import silence_tf
tf = silence_tf()

from lungai.data_extraction import get_data
from lungai.paths import TRAINED_MODELS_PATH, DATA_PATH
from lungai.ai import AI

dummy_path = join(TRAINED_MODELS_PATH, "dummy")

def pretty_print(label: str, confidence: float):
    print("Prediction: {label}, prob. = {confidence:.3%}".format(
            label=label, 
            confidence=confidence))


def test_load(silent=False):
    ai = AI.load(dummy_path)
    if not silent:
        ai.model.summary()
        print(ai.label_dict)
        print(ai.io_shape)

    return ai

def test_predict():
    ai = test_load(silent=True)

    data = get_data()[0]

    test_inds = [0, 1]

    predictions = [ ai.predict_one(item) for item in data[test_inds]]
    
    for label, confidence in predictions:
        pretty_print(label, confidence)
        
    

def test_eval_sound():
    ai = test_load(silent=True)
    record_name = "101_1b1_Al_sc_Meditron.wav"
    record_path = join(DATA_PATH, record_name)
    label, conf = ai.predict_sound(record_path)
    pretty_print(label, conf)

if __name__ == "__main__":
    test_eval_sound()
