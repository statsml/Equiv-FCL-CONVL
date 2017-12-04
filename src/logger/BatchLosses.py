import keras
class LossHistory(keras.callbacks.Callback):
    '''
    The class will store the loss for every batch during training.
    The using way is the same with the corresponding logger in Keras.
    '''
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))