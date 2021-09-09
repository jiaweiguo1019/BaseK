from basek.utils.tf_compat import tf, keras


class EmbeddingIndex(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super().build(input_shape)

    def call(self, index):
        self.index = index
        index = list(range(self.index))
        return tf.constant(index, dtype=tf.int64)

    def get_config(self, ):
        config = {'index': self.index}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
