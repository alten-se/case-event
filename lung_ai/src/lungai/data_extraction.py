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

def path_parrent(path: str, n=1) -> str:
    """ returns the nth parrent dir of path
    Examples:
       >>> path_parrent(a_path, 1) == os.path.dirname(a_path)
    True

       >>> path_parrent("path/to/file.txt", 2) == "path"
    True

       >>> path_parrent("path/to/file.txt", 0) == "path/to/file"
    True

    Args:
        path (str): path/to/a/file.txt or path/to/a/folder/

        n (int, optional): the amount 'parrent-levels' up to return. Defaults to 1.

    Returns:
        str: path of nth parrent
    """

    if n<1:
        return path
    n -= 1
    return path_parrent(os.path.dirname(path), n) 
    

PACKAGE_ROOT_PATH = path_parrent(__file__, 3)
DB_PATH = os.path.join(PACKAGE_ROOT_PATH, "db")

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
    src_cash_path = _get_cashe_paths()[3]

    if os.path.exists(src_cash_path):
        cash_diff = os.popen(" ".join(("diff", __file__, src_cash_path))).read()
    else:
        cash_diff = "{__file__} is not chashed"

    return bool(cash_diff)

def _get_cashe_paths() -> Tuple[str, str, str]:
    cashe_path = os.path.join(DB_PATH, "extract_cashe")
    my_name = os.path.basename(__file__)
    
    x_path = os.path.join(cashe_path, "x")
    y_path = os.path.join(cashe_path, "y")
    dict_path = os.path.join(cashe_path, "label.dict")
    src_cash_path = os.path.join(cashe_path, my_name)

    return x_path, y_path, dict_path, src_cash_path

def _load_cashe() -> Tuple[ndarray, ndarray, Dict]:
    x_path, y_path, dict_path = _get_cashe_paths()[:3]
    x = np.load(x_path + ".npy", allow_pickle=True)
    y = np.load(y_path + ".npy")
    with open(dict_path, "rb") as file:
        label_dict = pickle.load(file)
    return x, y, label_dict

def _save_cashe(x: ndarray, y: ndarray, label_dict: Dict):
    x_path, y_path, dict_path, src_cash_path = _get_cashe_paths()
    
    np.save(x_path, arr=x)
    np.save(y_path, arr=y)
    with open(dict_path, "wb+") as file:
        pickle.dump(label_dict, file)

    with open(src_cash_path, "w+") as cashe_file:
        with open(__file__, "r") as src:
            cashe_file.write(src.read())
