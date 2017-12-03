from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.utils import plot_model
import src.Data as Data
import src.net as net

(x_train, y_train),(x_test, y_test),input_shape, batch_size, num_classes, epoches =  Data.getMiniData()
np.random.seed(14343)
cnn = net.cnnOneLayer(input_shape, 2, (4, 4), batch_size=128)
plot_model(cnn, show_shapes=True, show_layer_names=True,to_file='./logs/cnn.pdf')
np.random.seed(14343)
fcnn =  net.fcOneLayer((13 * 13, 16), batch_size=128)
plot_model(fcnn, show_shapes=True, show_layer_names=True,to_file='./logs/fc.pdf')

K.clear_session()