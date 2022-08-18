from argparse import Namespace
from termios import N_MOUSE
import numpy as np
import librosa
import os
import pickle
import pandas as pd
import functools

from typing import Dict, Tuple
from numpy import ndarray
 
from lungai.augmentation import add_noise, shift, stretch
from lungai.paths import DB_PATH, DATA_CASHE_PATH

paths = Namespace(
    x = os.path.join(DATA_CASHE_PATH, "x"),
    y = os.path.join(DATA_CASHE_PATH, "y"),
    dict = os.path.join(DATA_CASHE_PATH, "label.dict"),
    src_cashe = os.path.join(DATA_CASHE_PATH, os.path.basename(__file__))
)

def extract_data(data_path: str, labels_path:str) -> Tuple[ndarray, ndarray, Dict]:
    """Extract feature from the Sound data. We extracted Mel-frequency cepstral coefficients( spectral
        features ), from the audio data. Augmentation of sound data by adding Noise, streaching and shifting
        is also implemented here. 40 features are extracted from each audio data and used to train the model.

    Args:
        data_path (str): path of folder containing wav files
        labels_path (str): path to csv of data labels

    Returns:
        X_: Array of features extracted from the sound file.
        y_: Array of target Labels.
    """
    x = []
    y = []
    copd_patients = []

    patient_disease_list = pd.read_csv(labels_path, sep=";")

    disease_dict = {}
    disease_counter = 0

    def get_disease(file: str) -> str:
        patient_id = file[:3]

        current_row = patient_disease_list.loc[patient_disease_list['patient_id'] == int(
            patient_id)]
        return current_row['disease'].values[0]

    rare_diseases = ("LRTI", "ASTHMA")

    sound_files = [f for f in os.listdir(data_path) if f[-3:] == "wav"]

    files = [f for f in sound_files if get_disease(
        f) not in rare_diseases][:100]

    print("Extracting data from n files:", len(files))

    for file in files:
        sound_path = os.path.join(data_path, file)
        current_disease = get_disease(file)

        if current_disease not in disease_dict.keys():
            disease_dict[current_disease] = disease_counter
            disease_counter += 1

        patient_id = file[:3]
        if (current_disease == 'COPD'):
            if sum([patient_id == p for p in copd_patients]) < 2:
                data_x, sampling_rate = librosa.load(
                    sound_path, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(
                    y=data_x, sr=sampling_rate, n_mfcc=40)
                copd_patients.append(patient_id)
                x.append(mfccs.T)
                y.append(disease_dict[current_disease])
        else:
            data_x, sampling_rate = librosa.load(
                sound_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40)

            def no_mod(x):
                return x
            noise_mod = functools.partial(add_noise, x=0.005)
            shift_mod = functools.partial(shift, x=1600)
            stretch_mod1 = functools.partial(stretch, rate=1.2)
            stretch_mod2 = functools.partial(stretch, rate=0.9)

            for aug in [no_mod, noise_mod, shift_mod, stretch_mod1, stretch_mod2]:
                modded_data = aug(data_x)
                modded_mfccs = librosa.feature.mfcc(
                    y=modded_data, sr=sampling_rate, n_mfcc=40)
                x.append(modded_mfccs.T)
                y.append(disease_dict[current_disease])

    x = np.array(x)
    y = np.array(y)

    return x, y, disease_dict

def get_data():
    if not cashe_diff():
        print("Cashe is valid, using chached data_extraction")
        x, y, label_dict = _load_cashe()
        return x, y, label_dict
    
    print("Cashe is invalid, redoing data extration...")

    data_path = os.path.join(DB_PATH, "data")
    labels_path = os.path.join(DB_PATH, "IBCHI_Challenge_diagnosis_v02.csv")
    x, y, label_dict = extract_data(data_path, labels_path)  # TODO save/load np.arrays
    
    _save_cashe(x, y, label_dict)
    return x, y, label_dict

def cashe_diff() -> bool:

    if os.path.exists(paths.src_cashe):
        cash_diff = os.popen(" ".join(("diff", __file__, paths.src_cashe))).read()
    else:
        cash_diff = "{__file__} is not chashed"

    return bool(cash_diff)



def _load_cashe() -> Tuple[ndarray, ndarray, Dict]:
    x = np.load(paths.x + ".npy", allow_pickle=True)
    y = np.load(paths.y + ".npy")
    with open(paths.dict, "rb") as file:
        label_dict = pickle.load(file)
    return x, y, label_dict

def _save_cashe(x: ndarray, y: ndarray, label_dict: Dict):
    np.save(paths.x, arr=x)
    np.save(paths.y, arr=y)
    with open(paths.dict, "wb+") as file:
        pickle.dump(label_dict, file)

    with open(paths.src_cashe, "w+") as cashe_file:
        with open(__file__, "r") as src:
            cashe_file.write(src.read())
