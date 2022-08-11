from operator import imod
from train import *
from data_extraction import load_data
import os

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

x, y, label_dict = load_data(wav_path, labels_path)
print("x shape:", x.shape, x.size)

print("## lables info")
for k, v in label_dict.items():
    print("- condition:", k, ", class_id:", v, ", count:", (y==v).sum())

model = rnn_model(input_shape=x.shape[1:], n_classes=len(label_dict))
trained_model = train(x, y, model)

print("Done!")
