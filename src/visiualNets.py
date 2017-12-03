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
print(cnn.get_layer('conv').output_shape)
# print(cnn.get_layer('act2').output_shape)

# print(cnn.get_layer('conv').output_shape)
np.random.seed(14343)
fcnn =  net.fcOneLayer((13 * 13, 16), batch_size=128)
# plot_model(fcnn, show_shapes=True, show_layer_names=True,to_file='./logs/fc.pdf')
print(fcnn.get_layer('conv').get_weights()[0].shape)
# print(fcnn.get_layer('conv').output_shape)
# print(fcnn.get_layer('act2').output_shape)

K.clear_session()