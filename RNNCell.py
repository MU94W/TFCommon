import tensorflow as tf
try:
    from tensorflow.python.ops.rnn_cell_impl import RNNCell
except ImportError:
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell


# Modified from "https://github.com/teganmaharaj/zoneout/blob/master/zoneout_seq2seq.py"
# Wrapper for the TF RNN cell
class ZoneoutWrapper(RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    def __init__(self, cell, state_zoneout_prob, training=True, seed=None, name="zoneout_wrapper"):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if (isinstance(state_zoneout_prob, float) and
                not (state_zoneout_prob >= 0.0 and state_zoneout_prob <= 1.0)):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                             % state_zoneout_prob)
        self._cell = cell
        if isinstance(self._cell.state_size, tuple):
            self._zoneout_prob = tuple([state_zoneout_prob]*len(self._cell.state_size))
        else:
            self._zoneout_prob = state_zoneout_prob
        self._seed = seed
        self.is_training = training
        self._name = name

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.name_scope(self.name):
            if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
                raise TypeError("Subdivided states need subdivided zoneouts.")
            if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
                raise ValueError("State and zoneout need equally many parts.")
            output, new_state = self._cell(inputs, state)
            if isinstance(self.state_size, tuple):
                if self.is_training:
                    new_state = tuple((1 - state_part_zoneout_prob) * tf.nn.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                                    for new_state_part, state_part, state_part_zoneout_prob in
                                    zip(new_state, state, self._zoneout_prob))
                else:
                    new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                    for new_state_part, state_part, state_part_zoneout_prob in
                                    zip(new_state, state, self._zoneout_prob))
            else:
                if self.is_training:
                    new_state = (1 - self._zoneout_prob) * tf.nn.dropout(
                        new_state - state, (1 - self._zoneout_prob), seed=self._seed) + state
                else:
                    new_state = self._zoneout_prob * state + (1 - self._zoneout_prob) * new_state
            return output, new_state


class AttentionWrapper(RNNCell):
    def __init__(self, cell, attention, name="attention_wrapper"):
        self._cell = cell
        self._attention = attention
        self._name = name
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope("AttentionWrapperZeroState"):
            return tuple([self._cell.zero_state(batch_size, dtype),
                          self._attention.zero_state(batch_size, dtype)])
    
    def __call__(self, inputs, state_tm1, scope=None):
        with tf.name_scope(self.name):
            rnn_state_tm1, att_state_tm1 = state_tm1

            inputs_context_tm1 = tf.concat([inputs, att_state_tm1["context"]], axis=-1)
            
            rnn_out_t, rnn_state_t = self._cell(inputs_context_tm1, rnn_state_tm1)

            context_t, att_state_t = self._attention(rnn_out_t, att_state_tm1)

            output_t = tf.concat([rnn_out_t, context_t], axis=-1)

            return output_t, tuple([rnn_state_t, att_state_t])



"""
或许不应该将多个U矩阵拼接在一起用orthogonal初始化。
"""
class GRUCell(RNNCell):
    """Gated Recurrent Unit"""
    def __init__(self, num_units, name="gru"):
        self._num_units = num_units
        self._gate_activation = tf.sigmoid
        self._name = name

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(self.name):

            # Check if the input size exist.
            input_size = x.shape.with_rank(2)[1].value
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            ### get weights.
            W_shape = (input_size, self.output_size)
            U_shape = (self.output_size, self.output_size)
            b_shape = (self.output_size,)
            Wrz = tf.get_variable(name="Wrz", shape=(input_size, 2 * self.output_size))
            Wh = tf.get_variable(name='Wh', shape=W_shape)
            Ur = tf.get_variable(name="Ur", shape=U_shape, initializer=tf.orthogonal_initializer())
            Uz = tf.get_variable(name="Uz", shape=U_shape, initializer=tf.orthogonal_initializer())
            Uh = tf.get_variable(name='Uh', shape=U_shape,
                                 initializer=tf.orthogonal_initializer())
            brz = tf.get_variable(name="brz", shape=(2 * self.output_size),
                                  initializer=tf.constant_initializer(0.))
            bh = tf.get_variable(name='bh', shape=b_shape,
                                 initializer=tf.constant_initializer(0.))

            ### calculate r and z
            rz_x = tf.matmul(x, Wrz) + brz
            r_x, z_x = tf.split(rz_x, num_or_size_splits=2, axis=1)
            r = self._gate_activation(r_x + tf.matmul(h_prev, Ur))
            z = self._gate_activation(z_x + tf.matmul(h_prev, Uz))

            ### calculate candidate
            h_slash = tf.tanh(tf.matmul(x, Wh) + tf.matmul(r * h_prev, Uh) + bh)

            ### final cal
            new_h = (1-z) * h_prev + z * h_slash

            return new_h, new_h



"""
或许不应该将多个U矩阵拼接在一起用orthogonal初始化。
"""
class LSTMCell(RNNCell):
    """Long Short-Term Memory (LSTM) unit recurrent network cell."""

    def __init__(self, num_units, forget_bias=1.0, name="lstm", scope=None):
        self._num_units = num_units
        self._gate_activation = tf.sigmoid
        self._forget_bias = forget_bias
        self._name = name
        _scope = name if scope is None else scope+"/"+name
        with tf.variable_scope(_scope, reuse=tf.AUTO_REUSE):
            u_shape = (self.output_size, self.output_size)
            mat_u_i = tf.get_variable(name="recurrent_kernel_i", shape=u_shape, initializer=tf.orthogonal_initializer())
            mat_u_o = tf.get_variable(name="recurrent_kernel_o", shape=u_shape, initializer=tf.orthogonal_initializer())
            mat_u_j = tf.get_variable(name="recurrent_kernel_j", shape=u_shape, initializer=tf.orthogonal_initializer())
            mat_u_f = tf.get_variable(name="recurrent_kernel_f", shape=u_shape, initializer=tf.orthogonal_initializer())
            self.mat_u = tf.concat([mat_u_i, mat_u_o, mat_u_j, mat_u_f], axis=-1)

    @property
    def state_size(self):
        return tuple([self.output_size, self.output_size])

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, state_prev, scope=None):
        with tf.variable_scope(self.name):
            h_prev, c_prev = state_prev

            # Check if the input size exist.
            input_size = x.shape.with_rank(2)[1].value
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            # get weights for concatenated tensor.
            mat_w = tf.get_variable(name='input_kernel', shape=(input_size, self.output_size*4))
            b = tf.get_variable(name='bias', shape=(self.output_size*4),
                                initializer=tf.constant_initializer(0.))

            # calculate gates and input's info.
            i_o_j_f_x = tf.matmul(x, mat_w) + b
            i_o_j_f_h = tf.matmul(h_prev, self.mat_u)
            i_o_j_f = i_o_j_f_x + i_o_j_f_h
            i, o, j, f = tf.split(i_o_j_f, num_or_size_splits=4, axis=-1)

            # activate them!
            i, o = tf.tanh(i), self._gate_activation(o)
            j, f = self._gate_activation(j), self._gate_activation(f + self._forget_bias)

            # calculate candidate.
            new_c = f * c_prev + j * i

            # final cal.
            new_h = o * tf.tanh(new_c)

            return new_h, tuple([new_h, new_c])


class PreDNNWrapper(RNNCell):
    def __init__(self, cell, dnn_fn, name="pre_projection"):
        self._cell = cell
        self._dnn_fn = dnn_fn
        self._name = name
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state_tm1, scope=None):
        with tf.name_scope(self.name):
            dnn_out = self._dnn_fn(inputs)
            rnn_out_t, rnn_state_t = self._cell(dnn_out, state_tm1)

            return rnn_out_t, rnn_state_t


class PostDNNWrapper(RNNCell):
    def __init__(self, cell, dnn_fn, name="post_projection"):
        self._cell = cell
        self._dnn_fn = dnn_fn
        self._name = name
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state_tm1, scope=None):
        with tf.name_scope(self.name):
            rnn_out_t, rnn_state_t = self._cell(inputs, state_tm1)
            dnn_out = self._dnn_fn(rnn_out_t)

            return dnn_out, rnn_state_t
