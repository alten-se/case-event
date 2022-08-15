from typing import Tuple
from numpy import ndarray

import numpy as np
import tensorflow as tf
from keras.models import Sequential

def train(x: ndarray, y:ndarray, model: Sequential) -> Sequential:
    '''
        Training the Neural Network model against the data.
        Args:
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''
    batch_size = x.shape[0]

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # TODO k-fold cross validation, train diffrent instaseces using diffrent selections for train/validate set

    train_set, validate_set = split_data(x, y)

    for patient_class in np.unique(validate_set[1]):
        print("cv:", sum(validate_set[1]==patient_class), "ct:", sum(train_set[1]==patient_class), "frac:", sum(validate_set[1]==patient_class)/(sum(validate_set[1]==patient_class) + sum(train_set[1]==patient_class)))

    model.fit(train_set[0], train_set[1], epochs=500, validation_data=validate_set)


    return model

DataSet = Tuple[ndarray, ndarray]

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
