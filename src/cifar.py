'''
The code is based on keras example Cifar10-CNN
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Conv2D


def cnnOneLayer(input_shape,stride,kernel_size,padding='valid',batch_size=None):
    model = Sequential()
    model.add(Conv2D(128, kernel_size,strides=stride, padding=padding,
                     input_shape=input_shape,
                     batch_size=batch_size,
                     kernel_initializer='RandomUniform',use_bias=False, name='conv'))
    return model
