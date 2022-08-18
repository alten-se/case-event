import numpy as np
from os.path import join as p_join

from lungai.load_model import load_trained_model
from lungai.data_extraction import get_data
from lungai.paths import TRAINED_MODELS_PATH
from lungai.evaluate import eval


def test_load():
    data, labels, label_dict = get_data()
    model = load_trained_model(p_join(TRAINED_MODELS_PATH, "dummy"))

    test_inds = [0, 1]

    predictions = [ eval(data_point, model, label_dict) for data_point in data[test_inds]]
    
    for label, confidence in predictions:
        print("Prediction: {label}, prob. = {confidence:.3%}".format(
            label=label, 
            confidence=confidence))
    

if __name__ == "__main__":
    test_load()
