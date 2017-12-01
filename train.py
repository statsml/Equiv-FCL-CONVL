from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
from logger.BatchLosses import LossHistory
import pandas


def getMiniData():
    batch_size = 128
    num_classes = 10
    epoches = 1
    # input image dimensions
    img_rows, img_cols, channel = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], -1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], -1, img_rows, img_cols)
        input_shape = (channel, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, -1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, -1)
        input_shape = (img_rows, img_cols, channel)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test),input_shape,batch_size,num_classes,epoches


def trainMSE(model, x_train, y_train, x_test, y_test, logpath, epoches=12, batch_size=128):
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam())
    lossbatch  = LossHistory()

    history=model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs=epoches,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[lossbatch])

    pandas.DataFrame({'loss':lossbatch.losses,'acc':lossbatch.acc}).to_csv(logpath)
    return model,history

