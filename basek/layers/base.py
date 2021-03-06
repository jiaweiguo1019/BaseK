from basek.utils.imports import numpy as np

from basek.utils.tf_compat import keras, tf


BatchNormalization = keras.layers.BatchNormalization
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Lambda = keras.layers.Lambda
Layer = keras.layers.Layer
LayerNormalization = keras.layers.LayerNormalization
Input = keras.layers.Input


class BiasAdd(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.bias = self.add_weight(
            shape=[dim], initializer=keras.initializers.Zeros(), name='bias'
        )
        super().build(input_shape)

    def call(self, x):
        x = x + self.bias
        return x


class Concatenate(Layer):

    def __init__(self, axis, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def call(self, x):
        return tf.concat(x, axis=self.axis)


class Index(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def call(self):
        index = np.array(self.index).reshape(-1, 1)
        return tf.constant(index, dtype=tf.int64)

    def __call__(self):
        return self.call()

    def get_config(self, ):
        config = {'max_index': self.max_idx}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
