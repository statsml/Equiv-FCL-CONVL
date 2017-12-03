from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.models import Model

import im2col as conveter
import net
import Data
(x_train, y_train),(x_test, y_test),input_shape, batch_size, num_classes, epoches =  Data.getMiniData()
(x_train, y_train),(x_test, y_test) = (x_train[1001:2001,:,:,:], y_train[1001:2001]),(x_test[1001:2001,:,:,:], y_test[1001:2001]),
np.random.seed(14343)
cnn = net.cnnOneLayer(input_shape, 2, (4, 4))
cnn.load_weights('./model/cnn.h5')
cnn_conv = Model(inputs=cnn.input,
                                 outputs=cnn.get_layer('conv').output)
outcnn = cnn_conv.predict(x_train,batch_size=100,verbose=0)

W1 = cnn.get_layer('conv').get_weights()[0].flatten()
mean1 = np.mean(W1)
s = np.std(W1)
print('CNN weight matrix mean {}, std {}'.format(mean1, s))
# print(cnn.get_layer('conv').output_shape)
np.random.seed(14343)
fcnn =  net.fcOneLayer((13 * 13, 16))
fcnn.load_weights('./model/fcnn.h5')
fcnn_conv = Model(inputs=fcnn.input,
                                 outputs=fcnn.get_layer('conv').output)

W2 = fcnn.get_layer('conv').get_weights()[0].flatten()
mean1 = np.mean(W2)
s = np.std(W2)
print('CNN weight matrix mean {}, std {}'.format(mean1, s))

N,_,_,_ = x_train.shape
print(x_train.shape)
X_train = conveter.im2col(x_train, (128,4,4,1), 2, (1000, 13, 13, 128))
print(X_train.shape)
X_train = np.reshape(X_train, (1000,-1,16))
print(X_train.shape)
outfc = fcnn_conv.predict(X_train, batch_size=100, verbose=0)

difference = 0.0
for i in range(1000):
    # print(np.linalg.norm((outcnn[i].flatten() - outfc[i].flatten())))
    difference = difference + np.linalg.norm((outcnn[i].flatten() - outfc[i].flatten()))
print(difference/1000.0)
K.clear_session()

import numpy
from matplotlib import pyplot
bins = numpy.linspace(-1, 1, 200)
pyplot.rcParams.update({'font.size': 15})
pyplot.hist(W1, bins, alpha=0.5, label='CNN', color='y')
pyplot.hist(W2, bins, alpha=0.5, label='FC', color='k')
print(np.linalg.norm((W1.flatten() - W2.flatten())))
pyplot.legend(loc='upper right')
pyplot.show()

#SGD
#1.84765432613e-06
#2.12273e-7

#adam
#0.535982942939
#0.0742322

