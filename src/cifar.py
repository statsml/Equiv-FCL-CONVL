from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Conv2D


def cnnOneLayer(input_shape,stride,kernel_size,channels,padding='same',batch_size=None):
    model = Sequential()
    model.add(Conv2D(channels, kernel_size,strides=stride, padding=padding,
                     input_shape=input_shape,
                     batch_size=batch_size,
                     kernel_initializer='he_uniform', name='conv3_'))
    model.add(Activation('relu', name='act3_'))
    model.add(Reshape(input_shape))
    return model

def fcOneLayer(input_shape, batch_size=None):
    model = Sequential()
    model.add(Reshape((28*28,),input_shape=input_shape, batch_size=batch_size))
    model.add(Dense(28*28,
                    input_shape=(28*28,),
                     kernel_initializer='he_uniform', name='conv3_'))
    model.add(Activation('relu', name='act3_'))
    model.add(Reshape(input_shape))
    return model
