import numpy as np

from lungai.data_extraction import get_data
from lungai.model import rnn_model
from lungai.train import train
from lungai.data_split import split_data
from lungai.data_gen import DataGenerator

extract_data = False
data, labels, label_dict = get_data(extract_data)

data_shape = data[0].shape
# None means unknown, in this case that we let n_time_steps variate
input_shape = (None, data_shape[-1])
model = rnn_model(input_shape=input_shape, n_classes=len(label_dict))

print("x shape:", data_shape)
print("len(data)", len(data))
print("## lables info")
for k, v in label_dict.items():
    print("- condition:", k, ", class_id:", v, ", count:", (labels == v).sum())


train_set, validate_set = split_data(data, labels, fraction=0.3)

for patient_class in np.unique(validate_set[1]):
    print("cv:", sum(validate_set[1] == patient_class), "ct:", sum(train_set[1] == patient_class), "frac:", sum(
        validate_set[1] == patient_class) / (sum(validate_set[1] == patient_class) + sum(train_set[1] == patient_class)))
print("train_len: ", len(train_set[1]))
print("validate_len: ", len(validate_set[1]))

train_gen = DataGenerator(train_set, batch_size=32, shuffle=True)
validate_gen = DataGenerator(validate_set, batch_size=32, shuffle=True)

trained_model = train(train_gen, validate_gen, model)
trained_model.save_weights("lung_ai/trained_models/w_temp/w_temp")

print("Done!")
