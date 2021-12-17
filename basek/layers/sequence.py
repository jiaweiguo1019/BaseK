from numpy.core.fromnumeric import transpose
from basek.utils.tf_compat import tf, keras

from basek.layers.base import (
    BiasAdd, BatchNormalization,
    Concatenate,
    Dense, Dropout,
    Embedding,
    Flatten,
    Index,
    Lambda, Layer, LayerNormalization
)


def scaled_dot_product_attention(q, k, v, mask, causality=False):

    """
    q: (b, h, seq_q, d)
    k: (b, h, seq_k, d)
    v: (b, h, seq_v, d)
    mask: (b, 1, seq_q, seq_k) or (seq_q, seq_k)
    """

    attention_logits = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = attention_logits / tf.math.sqrt(dk)

    if causality:
        seq_q = tf.shape(q)[2]
        look_ahead_mask = tf.linalg.band_part(tf.ones((seq_q, seq_q)), -1, 0)
        mask = tf.minimum(mask, look_ahead_mask)

    scaled_attention_logits += (mask - 1) * 1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output


class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, causality=False, use_bias=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causality = causality

        self.wq = Dense(d_model, use_bias=use_bias)
        self.wk = Dense(d_model, use_bias=use_bias)
        self.wv = Dense(d_model, use_bias=use_bias)

        self.depth = d_model // num_heads

        self.dense = Dense(d_model, use_bias=use_bias)

    def call(self, q, k, v, mask):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        v = tf.transpose(v, (0, 2, 1, 3))

        attention_outputs = scaled_dot_product_attention(q, k, v, mask, self.causality)
        attention_outputs = tf.transpose(attention_outputs, (0, 2, 1, 3))
        attention_outputs = tf.reshape(attention_outputs, (batch_size, -1, self.d_model))

        outputs = self.dense(attention_outputs)

        return outputs


class FFN(Layer):

    def __init__(self, d_model, dff):
        super().__init__()
        self.dense1 = Dense(dff, 'relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        print('ffn', x)
        x = self.dense2(x)
        return x


class SDMShortEncoderLayer(Layer):
    def __init__(self, d_model, num_heads=2, ffn_hidden_unit=128, dropout=0.1, causality=False):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super().__init__()
        self.lstm_layer = keras.layers.LSTM(d_model, return_sequences=True, dropout=dropout)
        self.mha1 = MultiHeadAttention(d_model, num_heads, causality)
        self.mha2 = MultiHeadAttention(d_model, num_heads, False)

        self.ffn1 = FFN(d_model, ffn_hidden_unit)
        self.ffn2 = FFN(d_model, ffn_hidden_unit)
        self.ffn3 = FFN(d_model, ffn_hidden_unit)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.layernorm4 = LayerNormalization(epsilon=1e-6)
        self.layernorm5 = LayerNormalization(epsilon=1e-6)
        self.layernorm6 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.dropout5 = Dropout(dropout)
        self.dropout6 = Dropout(dropout)

    def call(self, inputs, mask, training=None):
        # mask (batch_size, seq_len)
        uid_emb, x = inputs
        seq_mask = mask[:, :, tf.newaxis]

        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
        # self-attention
        lstm_out = self.lstm_layer(x, training=training)  # （None, seq_len, d_model)
        lstm_out = lstm_out * seq_mask
        lstm_out = self.dropout1(lstm_out, training=training)
        lstm_out = self.layernorm1(lstm_out + x)
        out1 = self.ffn1(lstm_out)
        out1 = self.dropout2(out1, training=training)
        out1 = self.layernorm2(out1 + lstm_out)
        out1 = out1 * seq_mask

        att_out = self.mha1(out1, out1, out1, attention_mask)
        att_out = self.dropout3(att_out, training=training)
        att_out = self.layernorm3(out1 + att_out)  # (None, seq_len, d_model)
        out2 = self.ffn2(att_out)
        out2 = self.dropout4(out2, training=training)
        out2 = self.layernorm4(out2 + att_out)
        out2 = out2 * seq_mask

        target_att = self.mha2(uid_emb, out2, out2, attention_mask)
        target_att = self.dropout5(target_att, training=training)
        out3 = self.ffn3(target_att)
        out3 = self.dropout6(out3)
        out3 = self.layernorm6(out3 + target_att)

        short_emb = out3

        return short_emb


class SDMLongEncoderLayer(Layer):

    def __init__(self, d_model, num_heads=2, ffn_hidden_unit=128, dropout=0.1):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, causality=False)
        self.mha2 = MultiHeadAttention(d_model, num_heads, causality=False)
        self.mha3 = MultiHeadAttention(d_model, num_heads, causality=False)
        self.mha4 = MultiHeadAttention(d_model, num_heads, causality=False)
        self.ffn = FFN(d_model, ffn_hidden_unit)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.layernorm4 = LayerNormalization(epsilon=1e-6)
        self.layernorm5 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.dropout5 = Dropout(dropout)

    def call(self, inputs, mask, training=None):

        uid_emb, iid_seq_emb, cid_seq_emb, bid_seq_emb, ts_diff_seq_emb = inputs
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]

        target_atten_iid_out = self.mha1(uid_emb, iid_seq_emb, iid_seq_emb, attention_mask)
        target_atten_iid_out = self.dropout1(target_atten_iid_out, training=training)

        target_atten_cid_out = self.mha1(uid_emb, cid_seq_emb, cid_seq_emb, attention_mask)
        target_atten_cid_out = self.dropout2(target_atten_cid_out, training=training)

        target_atten_bid_out = self.mha1(uid_emb, bid_seq_emb, bid_seq_emb, attention_mask)
        target_atten_bid_out = self.dropout3(target_atten_bid_out, training=training)

        target_atten_ts_diff_out = self.mha1(uid_emb, ts_diff_seq_emb, ts_diff_seq_emb, attention_mask)
        target_atten_ts_diff_out = self.dropout4(target_atten_ts_diff_out, training=training)

        target_atten_out = target_atten_iid_out + target_atten_cid_out + target_atten_bid_out + target_atten_ts_diff_out
        out = self.ffn(target_atten_out)
        out = self.dropout5(out, training=training)
        out = self.layernorm5(out + target_atten_out)

        long_emb = out

        return long_emb


class SDMGateLayer(Layer):

    def __init__(self, d_model, ffn_hidden_unit=128):
        super().__init__()
        self.ffn = FFN(d_model, ffn_hidden_unit)


    def call(self, inputs):
        _, s, l = inputs
        x = tf.concat(inputs, axis=-1)
        logits = self.ffn(x)
        gate = tf.math.sigmoid(logits)
        y = gate * s + (1. - gate) * l
        return y
