from model import rnn_model
import tensorflow as tf
from numpy import ndarray
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
    split_index = (batch_size * 1) // 3  # use a 1/3 of data for validation

    train_set = x[split_index:], y[split_index:]
    validate_set = x[:split_index], y[:split_index]

    model.fit(train_set[0], train_set[1], epochs=300, validation_data=validate_set)


    return model
