import pandas as pd
import librosa
from soundfile import SoundFile
import os
import pandas as pd

from typing import Optional, Callable
from numpy import ndarray
 
from lungai.augmentation import aug_dict
from lungai.paths import DATA_PATH, DIAGNOSIS_PATH


disease_table = pd.read_csv(DIAGNOSIS_PATH, sep=";")

def get_diagnosis(id: int) -> str:
    current_row = disease_table.loc[disease_table['patient_id'] == id]
    return current_row['disease'].values[0]

def extract_mfccs(file: str | SoundFile, augmentation: Optional[Callable] = None) -> ndarray:
    """generates mfcc's from sound file

    Args:
        file (str | SoundFile): path to file or soundfile object
        augmentation (Optional): a function to modif

    Returns:
        ndarray: Mel-frequency cepstral coefficients
    """
    data, sample_rate = librosa.load(file, res_type="kaiser_fast")
    if augmentation is not None:
        data = augmentation(data)

    return librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T

def extract_data() -> pd.DataFrame:

    sound_files = [f for f in os.listdir(DATA_PATH) if f[-3:] == "wav"]

    data_dict = {
        "patient_id": [],
        "record_index": [],
        "location": [],
        "channels": [],
        "equipment": [],
        "diagnosis": [],
        "crackles": [],
        "wheezes": [],
        "mffcs": [],
        "augmentation": [],
    }

    n_files = len(sound_files)


    for i, file_name in enumerate(sound_files):
        print(f"extracting from file {i+1} of {n_files} files, {file_name}")
        for mod_name, mod_fn in aug_dict.items():
            id, record_index, chest_loc, daq_mode, device_name = file_name.split("_")
            data_dict["patient_id"].append(id := int(id))
            data_dict["record_index"].append(record_index)
            data_dict["location"].append(chest_loc)
            data_dict["channels"].append(daq_mode)
            data_dict["equipment"].append(device_name[:-4]) 
            data_dict["diagnosis"].append(get_diagnosis(id))

            sound_path = os.path.join(DATA_PATH, file_name)
            txt_path = sound_path[:-4] + ".txt"
            sound_notes = pd.read_csv(txt_path, sep="\t", header=None, names=["start", "end", "crackles", "wheezes"])   
            n_crackles = (sound_notes["crackles"]==1).sum()
            n_wheezes = (sound_notes["wheezes"]==1).sum()
            
            data_dict["crackles"].append(n_crackles) 
            data_dict["wheezes"].append(n_wheezes) 
            mffcs = extract_mfccs(sound_path, augmentation=mod_fn)

            data_dict["mffcs"].append(mffcs)
            data_dict["augmentation"].append(mod_name)

    data_table= pd.DataFrame(data_dict)

    data_table.sort_values(["patient_id"], inplace=True)
    data_table.reset_index(inplace=True, drop=True)

    return data_table
        

