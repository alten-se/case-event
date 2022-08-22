import pytest
from os.path import join

from lungai.tf_config import silence_tf
tf = silence_tf()

from lungai.data_extraction import get_data
from lungai.paths import TRAINED_MODELS_PATH, DATA_PATH
from lungai.ai import AI

dummy_path = join(TRAINED_MODELS_PATH, "dummy")

@pytest.fixture
def dummy():
    return AI.load(dummy_path)

@pytest.fixture
def data_set():
    return get_data()


def pretty_print(label: str, confidence: float):
    print("Prediction: {label}, prob. = {confidence:.3%}".format(
            label=label, 
            confidence=confidence))

def test_load(dummy: AI):
    ai = dummy
    ai.model.summary()
    print(ai.label_dict)
    print(ai.io_shape)


def test_predict(dummy: AI, data_set):
    data = data_set[0]
    test_inds = [0, 1]
    predictions = [ dummy.predict_one(item) for item in data[test_inds]]
    
    for label, confidence in predictions:
        pretty_print(label, confidence)
        

def test_eval_sound(dummy: AI):
    record_name = "101_1b1_Al_sc_Meditron.wav"
    record_path = join(DATA_PATH, record_name)
    label, conf = dummy.predict_sound(record_path)
    pretty_print(label, conf)

def test_train(dummy):
    pass
    

if __name__ == "__main__":

    test_eval_sound()
