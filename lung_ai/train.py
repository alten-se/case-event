import tensorflow as tf
from keras.models import Sequential

from data_gen import DataGenerator


def train(train_gen: DataGenerator, validate_gen: DataGenerator, model: Sequential) -> Sequential:
    '''
        Training the Neural Network model against the data.
        Args:
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            # I use ._decayed_lr method instead of .lr
            return optimizer._decayed_lr(tf.float32)
        return lr

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3)
    lr_metric = get_lr_metric(opt)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", lr_metric]
    )

    model.fit(train_gen, validation_data=validate_gen, epochs=200)
    return model
