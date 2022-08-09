from train import *
from featureExtraction import *
import os

#current_dir = os.getcwd()
#    return current_dir + "\\" + path + "\\" + filename
my_folder = os.path.pardir(__file__)
data_folder = "Data" 
labels_path = os.path.join(
    my_folder, data_folder, "IBCHI_Challenge_diagnosis_v02.csv"
)
wav_path = os.path.join(
    my_folder, data_folder, "data"
)

X, y = InstantiateAttributes(wav_path, labels_path)

print(X.ndim)
# 1

print(type(X.ndim))
# <class 'int'>

print(X.ndim)
# 2

print(X.ndim)
# 3

print(X.shape)
# (3,)

print(type(X.shape))
# <class 'tuple'>

print(X.shape)
# (3, 4)

print(X.shape)
# (2, 3, 4)

print(X.size)
# 3

print(type(X.size))
# <class 'int'>

print(X.size)
# 12

print(X.size)
# 24

trainModel(X, y)
