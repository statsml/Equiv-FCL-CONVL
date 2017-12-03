from __future__ import print_function
import keras
from keras.callbacks import TensorBoard
from logger.BatchLosses import LossHistory
import pandas


def trainMSE(model, x_train, y_train, x_test, y_test, logpath, modelpath,historypath, epoches=12, batch_size=128 , names='cnn'):
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam())
    lossbatch  = LossHistory()
    tensorboard = TensorBoard(log_dir='./logs/'+names, histogram_freq=0,
                              write_graph=True, write_images=False)


    history=model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs=epoches,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[tensorboard,lossbatch])

    pandas.DataFrame({'loss':lossbatch.losses,'acc':lossbatch.acc}).to_csv(logpath)
    pandas.DataFrame(history.history).to_csv(historypath)
    model.save(modelpath,overwrite=True)
    return model,history
