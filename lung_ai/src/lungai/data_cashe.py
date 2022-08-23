import pandas as pd
import os

from lungai.paths import  DATA_CASHE_PATH
from lungai.data_extraction import extract_data

from lungai.data_extraction import __file__ as extration_file
from lungai.augmentation import __file__ as augmentation_file

TRACKED_SRC = {
    "cashe": __file__,
    "extraction": extration_file,
    "augmentation": augmentation_file
}

def get_cashe_path(src_path: str) -> str:
    return os.path.join(DATA_CASHE_PATH, os.path.basename(src_path))

CASHED_PATHES = {
    "data_table": os.path.join(DATA_CASHE_PATH, "table.pickle"),
    **{key: get_cashe_path(value) for key, value in TRACKED_SRC.items()}

}



def get_data(force_extract=False) -> pd.DataFrame:
    if (not cashe_diff()) and (not force_extract):
        print("Cashe is valid, using chached data_extraction")
        return pd.read_pickle(CASHED_PATHES["data_table"])
    
    print("Cashe is invalid, redoing data extration...")
    data_table = extract_data()
    _save_cashe(data_table)

    return data_table

def cashe_diff() -> bool:
    files_exists = all(os.path.exists(path) for path in CASHED_PATHES.values())
    if files_exists:
        diffs = [os.popen(" ".join(("diff", path, get_cashe_path(path)))).read() for path in TRACKED_SRC.values()]
        cash_diff= "".join(diffs)
    else:
        cash_diff = "Files is not chashed"

    return bool(cash_diff)

def _save_cashe(data_table: pd.DataFrame):
    data_table.to_pickle(CASHED_PATHES["data_table"], compression=None)

    for src_file in TRACKED_SRC.values():
        with open(src_file, mode="r") as read_file:
            with open(get_cashe_path(src_file), mode="w+") as write_file:
                write_file.write(read_file.read())

