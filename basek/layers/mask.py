from basek.utils.tf_compat import tf, keras


class MaskLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)