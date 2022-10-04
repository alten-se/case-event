from keras.layers import Input, Add, Dense, LeakyReLU
from keras.layers import CuDNNGRU as GRU
from keras.models import Model

from keras.utils import plot_model

in_shape = (None, 40)

in1 = Input(in_shape)

x1 = GRU(8)(in1)
x1 = LeakyReLU()(x1)

x2 = GRU(8)(GRU(10, return_sequences=True)(in1))


added = Add()((x1, x2))

out = Dense(4)(added)

model = Model(inputs=in1, outputs=out)

plot_model(model)

print(model)