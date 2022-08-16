from typing import Tuple
from numpy import ndarray

import numpy as np


DataSet = Tuple[ndarray, ndarray]

# TODO k-fold cross validation, train diffrent instaseces using diffrent selections for train/validate set
def split_data(data: ndarray, labels: ndarray, fraction = 0.3) -> Tuple[DataSet, DataSet]:
    """ Splits data set into training and validations sets.

    Ensures the same fraction is used for each class.

    Args:
        data (ndarray) shape: [batch_size, time_steps, features]
        labels (ndarray) : [batch_size, ]

    Returns:
        training and validation sets (Tuple[DataSet, DataSet]): [training_set, validation_set]
    """
    validate_mask = np.zeros(labels.shape, dtype=bool)
    classes = np.unique(labels)

    for c in classes:
        inds = np.nonzero(c == labels)[0]
        count_class = len(inds)
        validate_len = int(np.ceil((count_class*fraction)))
        validate_inds = inds[:validate_len]
        validate_mask[validate_inds] = True

    validate_data = data[validate_mask]
    validate_labels = labels[validate_mask] 

    train_data = data[~validate_mask]
    train_labels = labels[~validate_mask]

    return (train_data, train_labels), (validate_data, validate_labels)
