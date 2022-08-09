from train import *
from featureExtraction import *
import os

#current_dir = os.getcwd()
#    return current_dir + "\\" + path + "\\" + filename

path_root = os.getcwd() + '\\' + 'HE' + '\\' + 'Data'
path_wav_files = path_root + '\\' + 'ICBHI_final_database' + '\\'
path_patient_disease_list = path_root + \
    '\\' + 'IBCHI_Challenge_diagnosis_v02.csv'

X, y = InstantiateAttributes(path_wav_files, path_patient_disease_list)

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
