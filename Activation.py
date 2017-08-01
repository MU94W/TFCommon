import tensorflow as tf
from TFCommon.Initializer import gaussian_initializer, random_orthogonal_initializer

class Maxout(object):
    """Maxout activator - arXiv:1302.4389v4 [stat.ML] 20 Sep 2013
                        - Maxout Networks
    Args:
        
    """
    
    def __init__(self, units):
        self._units = units

    @property
    def units(self):
        return self._units

    def __call__(self, x, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Check if the input size exist.
            input_size = x.get_shape().with_rank(2)[1]
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")
            
            maxout_Wo = tf.get_variable(name='Wo', shape=(input_size, 2*self._units),
                                        initializer=gaussian_initializer(mean=0.0, std=0.01))
            maxout_b  = tf.get_variable(name='b',  shape=(2*self._units,),
                                        initializer=tf.constant_initializer(0.0))

            # 1st. Compute on all the 2 channels and reshape.
            t = tf.matmul(x, maxout_Wo) + maxout_b
            t = tf.reshape(t, shape=(-1, self._units, 2))

            # 2nd. Do maxout op, now has shape: (None, self._units)
            maxout_t = tf.reduce_max(t, axis=-1)

            return maxout_t

