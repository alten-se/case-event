import numpy as np
from os.path import join as p_join

from lungai.data_cashe import get_data
from lungai.data_split import split_data
from lungai.paths import TRAINED_MODELS_PATH
from lungai.ai import AI


def inspect_data(data, numeric_labels, label_dict):
    data_shape = data[0].shape
    print("x shape:", data_shape)
    print("len(data)", len(data))
    print("## lables info")
    for k, v in label_dict.items():
        print("- condition:", k, ", class_id:", v,
              ", count:", (numeric_labels == v).sum())


data_table = get_data()

data_array = data_table["mffcs"].values

unique_labels = data_table["diagnosis"].unique()

label_dict = {name: i for i, name in enumerate(unique_labels)}

numeric_labels = np.array([label_dict[name]
                          for name in data_table["diagnosis"]])


input_shape = (None, data_array[0].shape[-1])
n_classes = len(label_dict)

train_set, validate_set = split_data(data_array, numeric_labels, fraction=0.3)

inspect_data(data_array, numeric_labels, label_dict)

ai = AI(input_shape, n_classes)
ai.label_dict = label_dict

ai.train(train_set, validate_set, epochs=4)
output_path = p_join(TRAINED_MODELS_PATH, "latest")
ai.save(output_path)

print("Done!")
