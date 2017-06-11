import tensorflow as tf, numpy as np
from tensorflow.python.util import nest
if float(tf.__version__[:3]) >= 1.2:
    from tensorflow.python.ops.rnn_cell_impl import RNNCell
else:
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.keras import activations
from TFCommon.Initializer import gaussian_initializer, random_orthogonal_initializer
from six.moves import xrange

 
class GRUCell(RNNCell):
    """Gated Recurrent Unit (GRU) recurrent network cell."""

    def __init__(self, num_units, init_state=None, gate_activation="sigmoid", reuse=None):
        self.__num_units = num_units
        if gate_activation == "sigmoid":
            self.__gate_activation = tf.sigmoid
        elif gate_activation == "hard_sigmoid":
            self.__gate_activation = activations.hard_sigmoid
        else:
            raise ValueError
        self.__init_state = init_state
        self.__reuse = reuse

    @property
    def state_size(self):
        return self.__num_units

    @property
    def output_size(self):
        return self.__num_units

    def zero_state(self, batch_size, dtype):
        return tuple([super(GRUCell, self).zero_state(batch_size, dtype)])

    def init_state(self, batch_size, dtype):
        if self.__init_state is not None:
            return tuple([self.__init_state])
        else:
            return self.zero_state(batch_size, dtype)

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            h_prev = h_prev[0]

            # Check if the input size exist.
            input_size = x.shape.with_rank(2)[1].value
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            ### get weights.
            W_shape = (input_size, self.output_size)
            U_shape = (self.output_size, self.output_size)
            b_shape = (self.output_size,)
            Wz = tf.get_variable(name='Wz', shape=W_shape)
            Wr = tf.get_variable(name='Wr', shape=W_shape)
            Wh = tf.get_variable(name='Wh', shape=W_shape)
            Uz = tf.get_variable(name='Uz', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            Ur = tf.get_variable(name='Ur', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            Uh = tf.get_variable(name='Uh', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            bz = tf.get_variable(name='bz', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))
            br = tf.get_variable(name='br', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))
            bh = tf.get_variable(name='bh', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))

            ### calculate r and z
            r = self.__gate_activation(tf.matmul(x, Wr) + tf.matmul(h_prev, Ur) + br)
            z = self.__gate_activation(tf.matmul(x, Wz) + tf.matmul(h_prev, Uz) + bz)

            ### calculate candidate
            h_slash = tf.tanh(tf.matmul(x, Wh) + tf.matmul(r * h_prev, Uh) + bh)

            ### final cal
            new_h = (1-z) * h_prev + z * h_slash

            return new_h, tuple([new_h])

class LSTMCell(RNNCell):
    """Long Short-Term Memory (LSTM) unit recurrent network cell."""

    def __init__(self, num_units, gate_activation="sigmoid", forget_bias=1.0, reuse=None):
        self.__num_units = num_units
        if gate_activation == "sigmoid":
            self.__gate_activation = tf.sigmoid
        elif gate_activation == "hard_sigmoid":
            self.__gate_activation = activations.hard_sigmoid
        else:
            raise ValueError
        self.__forget_bias = forget_bias
        self.__reuse = reuse

    @property
    def state_size(self):
        return LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self.__num_units

    def __call__(self, x, state_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            h_prev, c_prev = state_prev

            # Check if the input size exist.
            input_size = x.shape.with_rank(2)[1].value
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            ### get weights for concated tensor.
            W_shape = (input_size, self.output_size)
            U_shape = (self.output_size, self.output_size)
            b_shape = (self.output_size,)
            Wi = tf.get_variable(name='Wi', shape=W_shape)
            Wj = tf.get_variable(name='Wj', shape=W_shape)
            Wf = tf.get_variable(name='Wf', shape=W_shape)
            Wo = tf.get_variable(name='Wo', shape=W_shape)
            Ui = tf.get_variable(name='Ui', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            Uj = tf.get_variable(name='Uj', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            Uf = tf.get_variable(name='Uf', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            Uo = tf.get_variable(name='Uo', shape=U_shape,
                                 initializer=random_orthogonal_initializer())
            bi = tf.get_variable(name='bi', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))
            bj = tf.get_variable(name='bj', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))
            bf = tf.get_variable(name='bf', shape=b_shape,
                                 initializer=tf.constant_initializer(self.__forget_bias)) # forget gate bias := 1
            bo = tf.get_variable(name='bo', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))

            ### calculate gates and input's info
            i = tf.tanh(tf.matmul(x, Wi) + tf.matmul(h_prev, Ui) + bi)
            j = self.__gate_activation(tf.matmul(x, Wj) + tf.matmul(h_prev, Uj) + bj)
            f = self.__gate_activation(tf.matmul(x, Wf) + tf.matmul(h_prev, Uf) + bf)
            o = tf.tanh(tf.matmul(x, Wo) + tf.matmul(h_prev, Uo) + bo)

            ### calculate candidate
            new_c = f * c_prev + i * j

            ### final cal
            new_h = o * tf.tanh(new_c)

            return new_h, LSTMStateTuple(new_h, new_c)

class AttentionWrapper(RNNCell):
    def __init__(self, cell, attention_module, reuse=None):
        self._cell = cell
        self._attention_module = attention_module
        self._reuse = reuse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def init_state(self, batch_size, dtype):
        return self._cell.init_state(batch_size, dtype)

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if isinstance(h_prev, tuple):
                h_prev_concated = tf.concat(h_prev, axis=-1)
            context, alpha = self._attention_module(h_prev_concated)
            output, new_h = self._cell(x, h_prev, context)
            
            return output, new_h, alpha, context

class AttentionWithoutIndicWrapper(AttentionWrapper):
    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if isinstance(h_prev, tuple):
                h_prev_concated = tf.concat(h_prev, axis=-1)
            context, alpha = self._attention_module(h_prev_concated)
            output, new_h = self._cell(context, h_prev)
            
            return output, new_h, alpha, context


