"""
Recurrent Neural Networks (RNN) - Deep Learning w/ Python, TensorFlow & Keras p.7
link = https://www.youtube.com/watch?v=BSpXCRTOLJA
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout # "CudNNLSTM" is GPU optimized variant of "LTSM"

print(tf.config.list_physical_devices())

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("CUDA enbabled GPU detected!")
    from keras.layers import CuDNNLSTM as LSTM
else:
    print("no cuda enbaled gpu, falling back to CPU")
    from keras.layers import LSTM


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = Sequential()

(time_steps, features) = x_train.shape[1:]

model.add(LSTM(
    128,
    input_shape=(x_train.shape[1:]),
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

print("training done")
