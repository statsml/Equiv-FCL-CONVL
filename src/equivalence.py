from __future__ import print_function
import cifar
import numpy as np
import Data as data
print('Prepare Data \n')
import im2col as conveter
from keras import backend as K
(x_train, y_train), (x_test, y_test), input_shape, batch_size, num_classes, epoches =  data.getMiniData()
x=x_train
y=y_train


cnneq = cifar.cnnOneLayer(input_shape, 2, (4,4))

error_algorithm1 = 0.0
error_mappingindex = 0.0
for j in range(1000):
    X = x_train[j:j+1]
    _, hout, wout, chout =  cnneq.get_layer('conv').output_shape
    cnnimgs = cnneq.predict(X,batch_size=1,verbose=0)
    # print(cnnimgs[0].shape)

    #Applying Algorithm 1 to compute CONV
    X_1 = conveter.im2col(X, (128,4,4,1), 2, (1, hout, wout, chout))
    # print(X_1.shape)
    W = cnneq.get_layer('conv').get_weights()[0]
    # print(W.shape)
    W_flatten = np.reshape(W, (-1,128))
    out_1 = np.dot(X_1, W_flatten)
    out_1 = np.reshape(out_1, (1, hout, wout, 128))
    error_algorithm1 = error_algorithm1 + np.linalg.norm(out_1.flatten() - cnnimgs.flatten())

    #Using mapping index to compute CONV
    X_mapindex = np.moveaxis(X, 3, 1)
    W_mapindex = cnneq.get_layer('conv').get_weights()[0]
    # print(W.shape)
    X_col = conveter.im2col_indices_(X_mapindex, 4, 4, padding=0, stride=2)
    W = W.transpose(3,2,0,1)
    W_col = np.reshape(W,(128, -1))
    # print(X_mapindex.shape)
    out_mapindex = np.dot(W_col,X_col)
    out_mapindex = np.reshape(out_mapindex,(128, hout, wout, -1))
    # print(out_mapindex.shape)
    out_mapindex = out_mapindex.transpose(3,1, 2,0)
    # print(out_mapindex.shape)
    error_mappingindex = error_mappingindex + np.linalg.norm(out_mapindex.flatten()-cnnimgs.flatten())

print(error_algorithm1/1000)
print(error_mappingindex/1000)
K.clear_session()
