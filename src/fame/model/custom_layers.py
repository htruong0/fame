import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class CoefSinhLayer(layers.Layer):
    def __init__(self, coef_scale, **kwargs):
        super(CoefSinhLayer, self).__init__(**kwargs)
        self.coef_scale = np.float64(coef_scale)
        
    def call(self, inputs):
        return tf.multiply(self.coef_scale, tf.math.sinh(inputs))
    
    def get_config(self):
        config = super().get_config()
        config["coef_scale"] = self.coef_scale
        return config

@tf.keras.utils.register_keras_serializable()
class LogLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(LogLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.log(inputs)
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class CoefLayer(layers.Layer):
    def __init__(self, coef_scale, **kwargs):
        super(CoefLayer, self).__init__(**kwargs)
        self.coef_scale = np.float64(coef_scale)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        constant = tf.constant(self.coef_scale)
        constant = tf.expand_dims(constant, axis=0)
        return tf.broadcast_to(constant, shape=(batch_size, 1))
        # return self.coef_scale
    
    def get_config(self):
        config = super().get_config()
        config["coef_scale"] = self.coef_scale
        return config