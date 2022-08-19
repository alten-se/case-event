import numpy as np
import pickle
from os.path import join as p_join

from lungai.data_extraction import get_data
from lungai.data_split import split_data
from lungai.paths import TRAINED_MODELS_PATH
from lungai.ai import AI

def inspect_data():
    data_shape = data[0].shape
    print("x shape:", data_shape)
    print("len(data)", len(data))
    print("## lables info")
    for k, v in label_dict.items():
        print("- condition:", k, ", class_id:", v, ", count:", (labels == v).sum())

    for patient_class in np.unique(validate_set[1]):
        print("cv:", sum(validate_set[1] == patient_class), "ct:", sum(train_set[1] == patient_class), "frac:", sum(
            validate_set[1] == patient_class) / (sum(validate_set[1] == patient_class) + sum(train_set[1] == patient_class)))
    print("train_len: ", len(train_set[1]))
    print("validate_len: ", len(validate_set[1]))


data, labels, label_dict = get_data()

input_shape = (None, data[0].shape[-1])
n_classes = len(label_dict)

train_set, validate_set = split_data(data, labels, fraction=0.3)

inspect_data()

ai = AI(input_shape, n_classes)
ai.label_dict = label_dict

ai.train(train_set, validate_set, epochs=200)
output_path = p_join(TRAINED_MODELS_PATH, "latest")
ai.save(output_path)

print("Done!")

