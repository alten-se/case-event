from model import rnn_model
import tensorflow as tf

def trainModel(x, y):
    '''
        Training the Neural Network model against the data.
        Args:
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''
    # batch_size = X.shape[0]
    # time_steps = X.shape[1]
    # data_dim = X.shape[2]
    (batch_size, time_steps, features) = x.shape

    model = rnn_model((time_steps, features))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # mc = ModelCheckpoint(
    #     'best_model.h5',
    #     monitor='val_acc',
    #     mode='auto',
    #     verbose=0,
    #     save_best_only=True)



    split_index = (batch_size * 1) // 3  

    train_set = x[split_index:], y[split_index:]
    validate_set = x[:split_index], y[:split_index]

    model.fit(train_set[0], train_set[1], epochs=10, validation_data=validate_set)


    return model
