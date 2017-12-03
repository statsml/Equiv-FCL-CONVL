from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
from logger.BatchLosses import LossHistory
import pandas
from functools import reduce


def trainMSE(model, x_train, y_train, x_test, y_test, logpath, modelpath,historypath, epoches=12, batch_size=128 , names='cnn'):
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam())
    lossbatch  = LossHistory()
    tensorboard = TensorBoard(log_dir='./logs/'+names, histogram_freq=0,
                              write_graph=True, write_images=False)

    # if config['denoise']:
    #     x_train = noise.addNoiseInput(x_train,config['sigma'])

    history=model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs=epoches,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[tensorboard,lossbatch])
    #print('====')
    #score = cifar3cnn.evaluate(x_test, y_test, verbose=0)
    pandas.DataFrame({'loss':lossbatch.losses,'acc':lossbatch.acc}).to_csv(logpath)
    pandas.DataFrame(history.history).to_csv(historypath)
    model.save(modelpath,overwrite=True)
    return model,history
