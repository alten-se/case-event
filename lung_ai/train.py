from typing import Tuple
from numpy import ndarray
from data_split import DataSet

import numpy as np
import tensorflow as tf
from keras.models import Sequential

from data_gen import DataGenerator




def train(train_set: DataSet, validate_set: DataSet, model: Sequential) -> Sequential:
    '''
        Training the Neural Network model against the data.
        Args:
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3)
    lr_metric = get_lr_metric(opt)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", lr_metric]
    )

    training_gen = DataGenerator(train_set[0], train_set[1], batch_size=32, shuffle=True)
    valid_gen = DataGenerator(validate_set[0], validate_set[1], batch_size=32, shuffle=True)


    for patient_class in np.unique(validate_set[1]):
        print("cv:", sum(validate_set[1]==patient_class), "ct:", sum(train_set[1]==patient_class), "frac:", sum(validate_set[1]==patient_class)/(sum(validate_set[1]==patient_class) + sum(train_set[1]==patient_class)))
    print("train_len: ", len(train_set[1]))
    print("validate_len: ", len(validate_set[1]))

    model.fit(training_gen, validation_data=valid_gen, epochs=200)
    return model

