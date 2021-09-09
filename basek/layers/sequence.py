from basek.utils.tf_compat import tf, keras


class SequencePoolingLayer(keras.layer.Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('mode must be sum, mean or max')
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        self.supports_masking = supports_masking
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, input, mask=None):
        if self.supports_masking:
            if mask is None:
                raise ValueError('When supports_masking=True,input must support masking')
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))