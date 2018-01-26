import tensorflow as tf

class Loss(object):
    def __init__(self):
        raise Exception('Unimplemented method')

    def __call__(self, labels, predictions):
        return self.loss_fn(labels, predictions)

class MSE_Loss(Loss):
    def __init__(self):
        self.loss_fn = tf.losses.mean_squared_error


        
