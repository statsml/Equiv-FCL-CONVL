from __future__ import print_function
import keras.backend as K
import net
import Data as data
import train as tr
import im2col as conveter
import numpy as np



(x_train, y_train),(x_test, y_test),input_shape, batch_size, num_classes, epoches =  data.getMiniData()
(x_train, y_train),(x_test, y_test) = (x_train[:1000,:,:,:], y_train[0:1000]),(x_test[0:1000,:,:,:], y_test[0:1000])
np.random.seed(14343)
cnn = net.cnnOneLayer(input_shape, 2, (4, 4))
# W1_cnn =  cnn.get_layer('conv').get_weights()[0]
# print(W1_cnn.shape)
# W2_cnn =  cnn.get_layer('dense').get_weights()[0]
# print(W2_cnn.shape)
#print(cnn.get_layer('conv').output_shape)
# print(cnn.get_layer('act2').output_shape)
tr.trainMSE(cnn, x_train, x_train,x_test, x_test,'./logs/cnn_batch.csv', './model/cnn.h5','./logs/cnn_history.csv',
            epoches=400, batch_size=128,
            names='cnn')
# print(cnn.get_layer('conv').output_shape)
np.random.seed(14343)
fcnn =  net.fcOneLayer((13 * 13, 16))

print(fcnn.get_layer('conv').get_weights()[0].shape)
print(fcnn.get_layer('dense').get_weights()[0].shape)
# w1new = np.reshape(W1_cnn,(-1, 128))

# print(w1new.shape)
# fcnn.get_layer('conv').set_weights((w1new,))
# fcnn.get_layer('dense').set_weights((W2_cnn,))
# print(fcnn.get_layer('conv').output_shape)
# print(fcnn.get_layer('act2').output_shape)

N,_,_,_ = x_train.shape
X_train = conveter.im2col(x_train, (128,4,4,1), 2, (1, 13, 13, 128))
X_test = conveter.im2col(x_test, (128,4,4,1), 2, (1, 13, 13, 128))
print(X_train.shape)
print(X_test.shape)
X_train = np.reshape(X_train, (1000,-1,16))
X_test = np.reshape(X_test, (1000,-1,16))

tr.trainMSE(fcnn,X_train,x_train,X_test,x_test,'./logs/fcnn_batch.csv', './model/fcnn.h5','./logs/fc_history.csv',
            epoches=400, batch_size=128,
            names='fc')


K.clear_session()