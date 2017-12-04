from __future__ import print_function
import keras.backend as K
import numpy as np
from keras.models import Model
import im2col as conveter
import net
import Data
#load data
(x_train, y_train),(x_test, y_test),input_shape, batch_size, num_classes, epoches =  Data.getMiniData()
#fetch 1000 images for F-norm
(x_train, y_train),(x_test, y_test) = (x_train[1001:2001,:,:,:], y_train[1001:2001]),(x_test[1001:2001,:,:,:], y_test[1001:2001]),

#load cnn and construct an intermidate model that can get the output of the first conv layer
cnn = net.cnnOneLayer(input_shape, 2, (4, 4))
cnn.load_weights('./model/cnn.h5')
cnn_conv = Model(inputs=cnn.input, outputs=cnn.get_layer('conv').output)
outcnn = cnn_conv.predict(x_train, batch_size=100, verbose=0)
#fetch the filters of the first conv layer in cnn
W1 = cnn.get_layer('conv').get_weights()[0].flatten()
#compute the mean and standard deviation of the filters.
mean1 = np.mean(W1)
s = np.std(W1)
print('CNN weight matrix mean {}, std {}'.format(mean1, s))

#load the fully connected network and construct an intermidate model that can get the output of the first dense layer in fc network.
fcnn =  net.fcOneLayer((13 * 13, 16))
fcnn.load_weights('./model/fcnn.h5')
fcnn_conv = Model(inputs=fcnn.input, outputs=fcnn.get_layer('conv').output)

#fecth the weights of the first dense layer in fc network.
W2 = fcnn.get_layer('conv').get_weights()[0].flatten()
#compute the mean and standard deviation of the weights
mean1 = np.mean(W2)
s = np.std(W2)
print('CNN weight matrix mean {}, std {}'.format(mean1, s))

#convert data to 2D matrices
N,_,_,_ = x_train.shape
print(x_train.shape)
X_train = conveter.im2col(x_train, (128,4,4,1), 2, (1000, 13, 13, 128))
print(X_train.shape)
X_train = np.reshape(X_train, (1000,-1,16))
print(X_train.shape)
outfc = fcnn_conv.predict(X_train, batch_size=100, verbose=0)

#compute F-norm of the outputs of the two intermidate models
difference = 0.0
for i in range(1000):
    difference = difference + np.linalg.norm((outcnn[i].flatten() - outfc[i].flatten()))
print(difference/1000.0)
K.clear_session()

#draw the hist figures of the filters and the weights
import numpy
from matplotlib import pyplot
bins = numpy.linspace(-1, 1, 200)
pyplot.rcParams.update({'font.size': 15})
pyplot.hist(W1, bins, alpha=0.5, label='CNN', color='y')
pyplot.hist(W2, bins, alpha=0.5, label='FC', color='k')
#compute F-norm between the filter and weights
print(np.linalg.norm((W1.flatten() - W2.flatten())))
pyplot.legend(loc='upper right')
pyplot.show()

