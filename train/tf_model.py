import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from tools import *

def build_CNN(n_classes, vocab_size, max_seq_length, channel_size=128, pool_size=2):
    X = keras.layers.Input((vocab_size, max_seq_length))
    out = Conv1D(filters=channel_size, kernel_size=3, strides=1, padding='same', activation='relu')(X)
    out = BatchNormalization()(out)
    out = MaxPooling1D(pool_size=pool_size)(out)
    out = Conv1D(filters=channel_size, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = BatchNormalization()(out)
    out = MaxPooling1D(pool_size=pool_size)(out)
    out = Flatten()(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(n_classes, activation='softmax')(out)
    model = Model(inputs=X, outputs=out)
    return model

