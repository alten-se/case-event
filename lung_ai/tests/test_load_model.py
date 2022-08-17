from lungai.model import rnn_model
from lungai.data_extraction import get_data

import os 
from keras.utils import pad_sequences
import numpy as np


TESTS_PATH = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.dirname(TESTS_PATH) 
TRAINED_AI_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

def test_load():
    data, labels, label_dict = get_data()
    data_shape = (None, data[0].shape[-1])
    model = rnn_model(data_shape, len(label_dict))
    model.load_weights(os.path.join(TRAINED_AI_PATH, "dummy/w"))

    test_inds = [0, 1]

    predictions = model.predict(pad_sequences(data[test_inds]))
    inv_map = {v: k for k, v in label_dict.items()}
    
    for p in predictions:
        index = np.argmax(p)
        print("Prediction: {condition}, prob. = {val:.3%}%".format(
            condition=inv_map[index], 
            val=p[index]))
    
    for index in test_inds:
        print("Truth: {}".format(inv_map[labels[index]]))


if __name__ == "__main__":
    test_load()
