from basek.utils.tf_compat import tf, keras


class EmbeddingIndex(keras.layers.Layer):

    def __init__(self, max_idx, **kwargs):
        self.max_idx = max_idx
        super().__init__(**kwargs)

    def call(self, index):
        index = list(range(self.max_idx))
        return tf.constant(index, dtype=tf.int64)

    def get_config(self, ):
        config = {'index': self.index}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
