import numpy as np
import librosa
from augmentation import add_noise, shift, stretch
import os
import pandas as pd
import functools


def load_data(dir_, path_patient_disease_list):
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

    def get_disease(id: str) -> str:
        current_row = patient_disease_list.loc[patient_disease_list['patient_id'] == int(id)]
        return current_row['disease'].values[0]
            
    files = [f for f in os.listdir(dir_) if f[-3:] =="wav"]
    print("Extracting data from n files:",len(files))
    
    def append_data(data, labels, mfccs, label):
        min_size = 700
        if mfccs.shape[1] < min_size:
            print("Warning discared a sound_entry due to tts small size:", mfccs.shape[1])
            return
        data.append(mfccs[:, :min_size].T)
        labels.append(label)

    for sound_file in files:
        patient_id = sound_file[:3] 
        sound_path = os.path.join(dir_, sound_file)
        current_disease = get_disease(patient_id)

        if current_disease.upper() in ("LRTI", "ASTHMA"):  # TODO Bronchictasis 
            # Do not use "Asthma" and "LRTI" since there are very few instances of those.
            continue

        if current_disease not in disease_dict.keys():
            disease_dict[current_disease] = disease_counter
            disease_counter += 1

        if (current_disease == 'COPD'):
            if sum([patient_id==p for p in copd_patients]) < 2:
                data_x, sampling_rate = librosa.load(sound_path, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40)
                copd_patients.append(patient_id)
                append_data(x, y, mfccs, disease_dict[current_disease])
        else:
            data_x, sampling_rate = librosa.load(sound_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40)

            no_mod = lambda x : x
            noise_mod = functools.partial(add_noise, x=0.005)
            shift_mod = functools.partial(shift, x=1600)
            stretch_mod1 = functools.partial(stretch, rate=1.2)
            stretch_mod2 = functools.partial(stretch, rate=0.9)

            for aug in [no_mod, noise_mod, shift_mod, stretch_mod1, stretch_mod2]:
                modded_data = aug(data_x)
                modded_mfccs = librosa.feature.mfcc(y=modded_data, sr=sampling_rate, n_mfcc=40)
                append_data(x, y, modded_mfccs, disease_dict[current_disease])

    x = np.array(x)
    y = np.array(y)

    return x, y, disease_dict
