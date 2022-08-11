from operator import imod
from train import *
from featureExtraction import *
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

x, y = InstantiateAttributes(wav_path, labels_path)
print("x shape:", x.shape, x.size)

model = rnn_model(input_shape=x.shape[1:], n_classes=7)
trained_model = train(x, y, model)

print("Done!")
