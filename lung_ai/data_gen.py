import numpy as np
from keras.utils import Sequence
from keras.utils import pad_sequences
from data_split import DataSet


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    Enable data sets of varing time length.
    __get_item__ Pads a batch of data to the same len before returning it
    """

    def __init__(self, data_set, batch_size=32, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.data = data_set[0]
        self.labels = data_set[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        inds = np.arange(index * self.batch_size, (index + 1) * self.batch_size)

        # Generate data
        x = self.data[inds]

        x = pad_sequences(x, value=0, dtype="float32")
        y = self.labels[inds]

        return x, y 

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            inds = np.arange(len(self.labels))
            np.random.shuffle(inds)
            self.data = self.data[inds]
            self.labels = self.labels[inds]

