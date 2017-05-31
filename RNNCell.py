import tensorflow as tf, numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from TFCommon.Initializer import gaussian_initializer, random_orthogonal_initializer
from six.moves import xrange

 
class GRUCell(RNNCell):
    """Gated Recurrent Unit Cell, which recept an additional peak or context tensor."""

    def __init__(self, num_units, init_state=None, reuse=None):
        self._num_units = num_units
        self._init_state = init_state
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def init_state(self, batch_size, dtype):
        if self._init_state is not None:
            return tuple([self._init_state])
        else:
            return tuple([self.zero_state(batch_size, dtype)])

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h_prev = h_prev[0]
            # Check if the input size exist.
            input_size = x.get_shape().with_rank(2)[1]
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            # Check num_units == state_size from h_prev.
            num_units = h_prev.get_shape().with_rank(2)[1]
            if num_units != self._num_units:
                raise ValueError("Shape of h_prev[1] incorrect: num_units %i vs %s" %
                                 (self._num_units, num_units))

            if num_units is None:
                raise ValueError("num_units from `h_prev` should not be None.")

            ### get weights for concated tensor.
            W_shape = (input_size, self._num_units)
            Wz = tf.get_variable(name='Wz', shape=W_shape,
                                 initializer=gaussian_initializer(mean=0.0, std=0.01))
            Wr = tf.get_variable(name='Wr', shape=W_shape,
                                 initializer=gaussian_initializer(mean=0.0, std=0.01))
            Wh = tf.get_variable(name='Wh', shape=W_shape,
                                 initializer=gaussian_initializer(mean=0.0, std=0.01))
            Uz = tf.get_variable(name='Uz', shape=(self._num_units, self._num_units),
                                 initializer=random_orthogonal_initializer())
            Ur = tf.get_variable(name='Ur', shape=(self._num_units, self._num_units),
                                 initializer=random_orthogonal_initializer())
            Uh = tf.get_variable(name='Uh', shape=(self._num_units, self._num_units),
                                 initializer=random_orthogonal_initializer())
            bz = tf.get_variable(name='bz', shape=(self._num_units,),
                                 initializer=tf.constant_initializer(0.0))
            br = tf.get_variable(name='br', shape=(self._num_units,),
                                 initializer=tf.constant_initializer(0.0))
            bh = tf.get_variable(name='bh', shape=(self._num_units,),
                                 initializer=tf.constant_initializer(0.0))

            ### calculate r and z
            r = tf.sigmoid(tf.matmul(x, Wr) + tf.matmul(h_prev, Ur) + br)
            z = tf.sigmoid(tf.matmul(x, Wz) + tf.matmul(h_prev, Uz) + bz)

            ### calculate candidate
            h_slash = tf.tanh(tf.matmul(x, Wh) + tf.matmul(r * h_prev, Uh) + bh)

            ### final cal
            new_h = (1-z) * h_prev + z * h_slash

            return new_h, tuple([new_h])


class ResidualWrapper(RNNCell):
    def __init__(self, cell, reuse=None):
        self._cell = cell
        self._init_state = self._cell._init_state
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
            # Check if input has the same shape as cell's output
            input_unit = x.get_shape().with_rank(2)[-1]
            if input_unit != self.output_size:
                raise ValueError("Shape of x (%d) is not equal to output_size (%d)" % (input_unit, self.output_size))

            output, new_h = self._cell(x, h_prev)
            output = output + tf.identity(x)
            
            return output, new_h


class DecoderWrapper(RNNCell):
    def __init__(self, cell, reuse=None):
        self._cell = cell
        self._init_state = self._cell._init_state
        self._reuse = reuse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def init_state(self, batch_size, dtype):
        return self._cell.init_state(batch_size, dtype)
    
    def __call__(self, x, h_prev, info, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            concated = tf.concat([x, info], axis=-1)
            output, new_h = self._cell(concated, h_prev)

            return output, new_h
            

class MultiCellWrapper(RNNCell):
    """Gated Recurrent Unit Cell, which recept an additional peak or context tensor."""

    def __init__(self, cells, reuse=None):
        if not isinstance(cells, list):
            cells = list([cells])
        self._cells = cells
        self._init_state = tuple([cell._init_state for cell in self._cells])
        self._reuse = reuse
        self._cell_num = len(cells)

    @property
    def state_size(self):
        sum_state = 0
        for item in self._cells:
            sum_state += item.state_size
        return sum_state

    @property
    def output_size(self):
        return self._cells[-1]._num_units

    @property
    def cell_num(self):
        return self._cell_num

    def init_state(self, batch_size, dtype):
        state_lst = tuple([cell.init_state(batch_size, dtype) for cell in self._cells])
        return state_lst

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            new_h_lst = []
            for idx in xrange(self._cell_num):
                with tf.variable_scope('cell_%d' % idx):
                    x, new_h = self._cells[idx](x, h_prev[idx])
                    new_h_lst.append(new_h)

            output = x
            return output, tuple(new_h_lst)


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


