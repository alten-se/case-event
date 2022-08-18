import os
from os.path import join

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
    
SRC_PATH = path_parrent(__file__)
PACKAGE_PATH = path_parrent(__file__, 3)
DB_PATH = join(PACKAGE_PATH, "db")
DATA_PATH = join(DB_PATH, "data")
TABLE_PATH = join(DB_PATH, "IBCHI_Challenge_diagnosis_v02.csv")


TRAINED_MODELS_PATH = join(PACKAGE_PATH, "trained_models")
DATA_CASHE_PATH = join(DB_PATH, "extract_cashe")