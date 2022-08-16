import numpy as np
import librosa
from augmentation import add_noise, shift, stretch
import os
import pickle
import pandas as pd
import functools


def extract_data(dir_, path_patient_disease_list):
    """Extract feature from the Sound data. We extracted Mel-frequency cepstral coefficients( spectral
        features ), from the audio data. Augmentation of sound data by adding Noise, streaching and shifting
        is also implemented here. 40 features are extracted from each audio data and used to train the model.

    Args:
        dir_ (_type_): path of folder containing wav files
        path_patient_disease_list (_type_): path to csv of data labels

    Returns:
        X_: Array of features extracted from the sound file.
        y_: Array of target Labels.
    """
    x = []
    y = []
    copd_patients = []

    patient_disease_list = pd.read_csv(path_patient_disease_list, sep=";")

    disease_dict = {}
    disease_counter = 0

    def get_disease(file: str) -> str:

        current_row = patient_disease_list.loc[patient_disease_list['patient_id'] == int(
            id)]
        return current_row['disease'].values[0]

    rare_diseases = ("LRTI", "ASTHMA")

    sound_files = [f for f in os.listdir(dir_) if f[-3:] == "wav"]

    files = [f for f in sound_files if get_disease(
        f) not in rare_diseases][:100]

    print("Extracting data from n files:", len(files))

    for file in files:
        sound_path = os.path.join(dir_, file)
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
                x.append[modded_mfccs.T]
                y.append[disease_dict[current_disease]]

    x = np.array(x)
    y = np.array(y)

    return x, y, disease_dict


def get_data(extract=False):
    my_folder = os.path.dirname(__file__)
    data_folder = "Data"
    labels_path = os.path.join(
        my_folder, data_folder, "IBCHI_Challenge_diagnosis_v02.csv"
    )
    wav_path = os.path.join(
        my_folder, data_folder, "data" + os.path.sep
    )

    x_path = os.path.join(my_folder, "Data", "out", "x")
    y_path = os.path.join(my_folder, "Data", "out", "y")
    label_path = os.path.join(my_folder, "Data", "out", "label.dict")
    if extract:
        x, y, label_dict = extract_data(
            wav_path, labels_path)  # TODO save/load np.arrays
        np.save(x_path, arr=x)
        np.save(y_path, arr=y)
        with open(label_path, "wb+") as file:
            pickle.dump(label_dict, file)
    else:
        x = np.load(x_path + ".npy", allow_pickle=True)
        y = np.load(y_path + ".npy")
        with open(label_path, "rb") as file:
            label_dict = pickle.load(file)

    return x, y, label_dict
