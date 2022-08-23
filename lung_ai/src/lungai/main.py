from typing import Tuple, Dict
import numpy as np
import pandas as pd
from os.path import join as p_join

from lungai.data_cashe import get_data
from lungai.data_split import split_data
from lungai.paths import TRAINED_MODELS_PATH
from lungai.ai import AI


def get_data_labels(data_table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = data_table["mffcs"].values
    label_column = "diagnosis"
    unique_labels = data_table[label_column].unique()
    label_dict = {name: i for i, name in enumerate(unique_labels)}
    numeric_labels = np.array([label_dict[name]
                          for name in data_table[label_column]])

    return data, numeric_labels, label_dict

def inspect_data(data_table: pd.DataFrame):

    data, numeric_labels, label_dict = get_data_labels(data_table)

    data_shape = data[0].shape
    print("x shape:", data_shape)
    print("len(data)", len(data))
    print("## lables info")
    total = len(numeric_labels)
    for k, v in label_dict.items():
        class_count = (numeric_labels == v).sum()
        fraction  = class_count/total
        print(f"- class_name: {k}, class_id:, {v}, count: {class_count}, fraction: {fraction:.2%}")


data_table = get_data()
for col in data_table.columns:
    print(col)

# lets look at unaugmented data

real_table = data_table.loc[data_table["augmentation"] == "no_mod"]

inspect_data(real_table)

rare_diseases = np.array(["Asthma", "LRTI"])[np.newaxis, :]

common_table = data_table.loc[(data_table["diagnosis"].values[:, np.newaxis] != rare_diseases).all(axis=1)]

inspect_data(common_table)

# dont use augmented data for COPD

balanced1 = common_table.loc[(common_table["diagnosis"] != "COPD") | (common_table["augmentation"] == "no_mod")]

data, numeric_labels, label_dict = get_data_labels(balanced1)

inspect_data(balanced1)

input_shape = (None, data[0].shape[-1])
n_classes = len(label_dict)

train_set, validate_set = split_data(data, numeric_labels, fraction=0.3)


ai = AI(input_shape, n_classes)
ai.label_dict = label_dict
ai.model.summary()

ai.train(train_set, validate_set, epochs=100)
output_path = p_join(TRAINED_MODELS_PATH, "latest")
ai.save(output_path)

print("Done!")
