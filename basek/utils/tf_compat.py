import os
import sys


if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def import_tf_compact():
    """Import tensorflow with compact behavior."""
    if 'tensorflow' not in sys.modules:
        # if 3 > int(tf.__version__.split('.')[0]) > 1:
        try:
            import tensorflow.compat.v1 as tf
            if tf.__version__ < '1.15.0':
                raise ImportError('tf version should not older than 1.15!')
            tf.logging.set_verbosity(tf.logging.ERROR)
            tf.disable_v2_behavior()
        except ImportError:
            raise ImportError('tf version should not older than 1.15!')
        return tf
    else:
        return sys.modules['tensorflow']


def import_keras():
    if 'keras' not in sys.modules:
        try:
            import keras
            if keras.__version__ < '2.3.0':
                raise ImportError('keras version should not older than 2.3!')
        except ImportError:
            raise ImportError('keras version should not older than 2.3!')
        return keras
    else:
        return sys.modules['keras']


tf = import_tf_compact()
try:
    keras = import_keras()
except ImportError:
    keras = tf.keras
