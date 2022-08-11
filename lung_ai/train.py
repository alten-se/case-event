from keras import backend as K
from model import InstantiateModel
from keras.models import Model
from keras.optimizer_v2.adamax import Adamax
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import numpy as np


def trainModel(x, y):
    '''
        Training the Neural Network model against the data.
        Args:
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''
    K.clear_session()
    # batch_size = X.shape[0]
    # time_steps = X.shape[1]
    # data_dim = X.shape[2]
    (batch_size, time_steps, data_dim) = x.shape

    Input_Sample = Input(shape=(time_steps, data_dim))
    Output_ = InstantiateModel(Input_Sample)
    Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

    Model_Enhancer.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adamax(),
        run_eagerly=True
        )

    # ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,
    #    restore_best_weights=False)

    mc = ModelCheckpoint(
        'best_model.h5',
        monitor='val_acc',
        mode='auto',
        verbose=0,
        save_best_only=True)

    # class_weights = class_weight.compute_sample_weight('balanced',
    #                                                 np.unique(y[:,0],axis=0),
    #                                                 y[:,0])

    split_index = 10
    train_set = x[split_index:], y[split_index:]
    validate_set = x[:split_index], y[:split_index]

    ModelHistory = Model_Enhancer.fit(x=train_set[0], y=train_set[1], batch_size=10, epochs=1,
                                      validation_data=validate_set,
                                      callbacks=[mc],
                                      # class_weight=class_weights,
                                      verbose=1)

    return ModelHistory
