import os
import pickle
import numpy as np

from data_extraction import load_data
from model import rnn_model
from train import train

my_folder = os.path.dirname(__file__)
data_folder = "Data" 
labels_path = os.path.join(
    my_folder, data_folder, "IBCHI_Challenge_diagnosis_v02.csv"
)
wav_path = os.path.join(
    my_folder, data_folder, "data"+os.path.sep
)

extract_data = False
x_path = os.path.join(my_folder, "Data", "out", "x")
y_path = os.path.join(my_folder, "Data", "out", "y")
label_path = os.path.join(my_folder, "Data", "out", "label.dict")
if extract_data: 
    x, y, label_dict = load_data(wav_path, labels_path) #TODO save/load np.arrays
    np.save(x_path, arr=x)
    np.save(y_path, arr=y)
    with open(label_path, "wb+") as file:
        pickle.dump(label_dict, file)
else:
    x = np.load(x_path + ".npy")
    y = np.load(y_path + ".npy")
    with open(label_path, "rb") as file:
        label_dict = pickle.load(file)




model = rnn_model(input_shape=x.shape[1:], n_classes=len(label_dict))

print("x shape:", x.shape, x.size)
print("## lables info")
for k, v in label_dict.items():
    print("- condition:", k, ", class_id:", v, ", count:", (y==v).sum())


trained_model = train(x, y, model)

print("Done!")
