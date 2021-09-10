from basek.utils.tf_compat import keras, tf


class BiasAdd(keras.layers.Layer):

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=[self.dim,], initializer=keras.initializers.Zeros(), name="bias"
        )
        super().build(input_shape)

    def call(self, x):
        x = x + self.bias
        return x


class Index(keras.layers.Layer):

    def __init__(self, max_idx, **kwargs):
        self.max_idx = max_idx
        super().__init__(**kwargs)

    def call(self):
        index = list(range(self.max_idx))
        return tf.constant(index, dtype=tf.int64)

    def __call__(self):
        return self.call()

    def get_config(self, ):
        config = {'max_index': self.max_idx}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
