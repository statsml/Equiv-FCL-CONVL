from __future__ import print_function
import keras.backend as K
import numpy as np
import im2col as conveter
import net
import train as tr
import os
import Data
#if the storing directories does not exist, make them
if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists('./model'):
        os.mkdir('./model')
#fetch data
(x_train, y_train),(x_test, y_test),input_shape, batch_size, num_classes, epoches =  Data.getMiniData()
#get 1000 images for training and validation respectively
(x_train, y_train),(x_test, y_test) = (x_train[:1000,:,:,:], y_train[0:1000]),(x_test[0:1000,:,:,:], y_test[0:1000])

#construct and train CNN
np.random.seed(14343)
cnn = net.cnnOneLayer(input_shape, 2, (4, 4))
#fetch intialization of CNN before training.
W1_cnn =  cnn.get_layer('conv').get_weights()[0]
print(W1_cnn.shape)
W2_cnn =  cnn.get_layer('dense').get_weights()[0]

tr.trainMSE(cnn, x_train, x_train,x_test, x_test,'./logs/cnn_batch.csv', './model/cnn.h5','./logs/cnn_history.csv',
            epoches=400, batch_size=128,
            names='cnn')

#construct and train fully connected network
np.random.seed(14343)
fcnn =  net.fcOneLayer((16,))
#load the initialzation from CNN's initialzation
w1new = np.reshape(W1_cnn,(-1, 128))
fcnn.get_layer('conv').set_weights((w1new,))
fcnn.get_layer('dense').set_weights((W2_cnn,))
#prepare data for training fully connected network
N,_,_,_ = x_train.shape
X_train = conveter.im2col(x_train, (128,4,4,1), 2, (1, 13, 13, 128))
X_test = conveter.im2col(x_test, (128,4,4,1), 2, (1, 13, 13, 128))
X_train = np.reshape(X_train, (1000,-1,16))
X_test = np.reshape(X_test, (1000,-1,16))

tr.trainMSE(fcnn,X_train,x_train,X_test,x_test,'./logs/fcnn_batch.csv', './model/fcnn.h5','./logs/fc_history.csv',
            epoches=400, batch_size=128,
            names='fc')


K.clear_session()