from train import *
from featureExtraction import *
import os

#current_dir = os.getcwd()
#    return current_dir + "\\" + path + "\\" + filename
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

model_history = trainModel(x, y)
