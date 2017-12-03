from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape,Flatten
from keras.layers import Conv2D

def cnnOneLayer_value(input_shape,stride,kernel_size,padding='valid',batch_size=None):
    model = Sequential()
    model.add(Conv2D(128, kernel_size,strides=stride, padding=padding,
                     input_shape=input_shape,
                     batch_size=batch_size,
                     kernel_initializer='he_uniform',use_bias=False, name='conv'))
    return model

def cnnOneLayer(input_shape,stride,kernel_size,padding='valid',batch_size=None):
    model = Sequential()
    model.add(Conv2D(128, kernel_size,strides=stride, padding=padding,
                     input_shape=input_shape,
                     batch_size=batch_size,
                     kernel_initializer='he_uniform',use_bias=False, name='conv'))
    model.add(Activation('relu', name='relu_conv'))
    model.add(Flatten(name='flaten'))
    model.add(Dense(28*28,
                     kernel_initializer='he_uniform',use_bias=False, name='dense'))
    model.add(Activation('relu', name='relu_dense'))
    model.add(Reshape(input_shape, name='reshape'))
    return model

def fcOneLayer(input_shape, batch_size=None):
    model = Sequential()
    model.add(Dense(128,
                    input_shape=input_shape,batch_size=batch_size,
                     kernel_initializer='he_uniform',use_bias=False, name='conv'))
    model.add(Activation('relu', name='relu_conv'))
    model.add(Flatten(name='flaten'))
    model.add(Dense(28 * 28,
                    kernel_initializer='he_uniform', use_bias=False, name='dense'))
    model.add(Activation('relu', name='relu_dense'))
    model.add(Reshape((28,28,-1), name='reshape'))
    return model
