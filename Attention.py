import tensorflow as tf


def mask_seq(inputs, sequence_length, name="mask_seq"):
    with tf.name_scope(name, values=[inputs, sequence_length]):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        inputs_dims = inputs.shape.ndims
        new_shape = [batch_size, time_steps] + [1] * (inputs_dims - 2)
        mask_seq = tf.reshape(tf.sequence_mask(sequence_length, time_steps, dtype=tf.float32), new_shape)
        return mask_seq * inputs


def bias_seq(inputs, sequence_length, bias=-1e5, name="bias_seq"):
    with tf.name_scope(name, values=[inputs, sequence_length]):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        inputs_dims = inputs.shape.ndims
        new_shape = [batch_size, time_steps] + [1] * (inputs_dims - 2)
        bias_seq = tf.reshape(tf.sequence_mask(sequence_length, time_steps, dtype=tf.bool), new_shape)
        bias_seq = bias * tf.cast(tf.logical_not(bias_seq), tf.float32)
        return bias_seq + inputs


class ContentAttention(object):
    """Attention Module
    Args:
        attention_units:    The attention module's capacity (should be proportional to query_units)
        memory:             A tensor, whose shape should be (None, Time, Unit)
        time_major:
    """
    def __init__(self, memory, sequence_length, units, alignments_history=True, name="ContentAttention"):
        self.memory = memory
        self.sequence_length = sequence_length
        self.units = units
        self.batch_size = tf.shape(memory)[0]
        self.enc_length = tf.shape(memory)[1]
        self.enc_units = memory.get_shape()[-1].value

        self.alignments_history = alignments_history
        self.name = name

        with tf.variable_scope(name):
            self.hidden_feats = tf.layers.dense(self.memory, units, activation=None, use_bias=True)
            self.e_range_seq = tf.tile(tf.expand_dims(tf.range(0, self.enc_length, delta=1, dtype=tf.int32), axis=0), [self.batch_size, 1])
            self.e_bias_zero = tf.zeros(shape=(self.batch_size, self.enc_length), dtype=tf.float32)
            self.e_bias_neg = tf.ones(shape=(self.batch_size, self.enc_length), dtype=tf.float32) * (-1e5)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(self.name+"ZeroState"):
            state = {"context": tf.zeros([batch_size, self.enc_units], dtype=dtype),
                     "alignments": tf.zeros([batch_size, self.enc_length], dtype=dtype)}
            if self.alignments_history:
                state.update({"alignments_history": tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                              "time_clock": tf.constant(0, dtype=tf.int32)})
            return state
    
    def __call__(self, query, state_tm1):
        with tf.variable_scope(self.name):
            Va = tf.get_variable(name='Va', shape=(self.units,),
                                 initializer=tf.constant_initializer(0.0))

            # 1st. compute query_feat (query's repsentation in attention module)
            query_feat = tf.expand_dims(tf.layers.dense(query, self.units,
                                                        activation=None, use_bias=False),
                                        axis=1) # (b,1,c)

            # 2nd. compute the energy for all time steps in encoder (element-wise mul then reduce)
            e = tf.reduce_sum(Va * tf.nn.tanh(self.hidden_feats + query_feat), axis=-1, keep_dims=False) # (b,e_t)

            # 3rd. compute the score
            if self.sequence_length is not None:
                e = bias_seq(e, self.sequence_length)
            alpha = tf.nn.softmax(e)    # (b,e_t)

            # 4th. get the weighted context from memory (element-wise mul then reduce)
            context = tf.expand_dims(alpha, -1) * self.memory
            context = tf.reduce_sum(context, axis=1) # (b,c)

            state_t = {"context": context, "alignments": alpha}

            if self.alignments_history:
                state_t.update({"alignments_history": state_tm1["alignments_history"].write(state_tm1["time_clock"], alpha),
                                "time_clock": tf.add(state_tm1["time_clock"], 1)})

            return context, state_t


class GMMAttention(object):
    """Attention Module
    Args:
        attention_units:    The attention module's capacity (should be proportional to query_units)
        memory:             A tensor, whose shape should be (None, Time, Unit)
        time_major:
    """
    def __init__(self, memory, sequence_length, units, alignments_history=True, init_kappa_pos=-1., name="GMMAttention"):
        self.memory = memory
        self.sequence_length = sequence_length
        self.units = units
        self.batch_size = tf.shape(memory)[0]
        self.enc_length = tf.shape(memory)[1]
        self.enc_units = memory.get_shape()[-1].value
        self.alignments_history = alignments_history
        self.init_kappa_pos = init_kappa_pos

        self.name = name

        with tf.variable_scope(name):
            self.tmp_l = tf.reshape(tf.cast(tf.range(0, self.enc_length), tf.float32), shape=(1, self.enc_length, 1))
            self.mask = tf.sequence_mask(sequence_length, self.enc_length, tf.float32) if sequence_length is not None else None

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(self.name+"ZeroState"):
            state = {"context": tf.zeros([batch_size, self.enc_units], dtype=dtype),
                     "alignments": tf.zeros([batch_size, self.enc_length], dtype=dtype),
                     "kappa": self.init_kappa_pos + tf.zeros(shape=(batch_size, self.units), dtype=dtype)}
            if self.alignments_history:
                state.update({"alignments_history": tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                              "time_clock": tf.constant(0, dtype=tf.int32)})
            return state

    def __call__(self, query, state_tm1):
        with tf.variable_scope(self.name):
            # 1st.
            rho_beta_kappa_t_hat = tf.layers.dense(query, 3*self.units, activation=None)
            rho_beta_kappa_t_hat_exp = tf.exp(rho_beta_kappa_t_hat)

            rho_t_hat_exp, beta_t_hat_exp, kappa_t_hat_exp = tf.split(rho_beta_kappa_t_hat_exp, num_or_size_splits=3, axis=-1)
            rho_t = rho_t_hat_exp
            beta_t = beta_t_hat_exp
            kappa_t = state_tm1["kappa"] + kappa_t_hat_exp

            # 2nd.
            tmp_rho = tf.expand_dims(rho_t, 1)
            tmp_beta = tf.expand_dims(beta_t, 1)
            tmp_kappa = tf.expand_dims(kappa_t, 1)
            tmp_l = self.tmp_l

            phi_t = tmp_rho * tf.exp(- tmp_beta * tf.square(tmp_kappa - tmp_l))     # (batch, enc_length, components)

            # 3rd. compute the score
            score = tf.reduce_mean(phi_t, -1)     # (batch, enc_length)
            if self.mask is not None:
                score = score * self.mask

            # 4th. get the weighted context from memory (element-wise mul then reduce)
            context = tf.expand_dims(score, -1) * self.memory
            context = tf.reduce_sum(context, axis=1)    # (batch, enc_units)

            state_t = {"context": context,
                       "alignments": score,
                       "kappa": kappa_t}

            if self.alignments_history:
                state_t.update({"alignments_history": state_tm1["alignments_history"].write(state_tm1["time_clock"], score),
                                "time_clock": tf.add(state_tm1["time_clock"], 1)})

            return context, state_t


class MultiHeadAttentionWrapper(object):
    def __init__(self, att_mod_list, name="MultiHeadAttentionWrapper"):
        self.att_mod_list = att_mod_list
        self.name = name
    
    def zero_state(self, batch_size, dtype):
        return [att_mod.zero_state(batch_size, dtype) for att_mod in self.att_mod_list]
    
    def __call__(self, query, state_lst_tm1, reuse=tf.AUTO_REUSE):
        with tf.name_scope(self.name):
            context_lst = []
            state_lst_t = []
            for att_mod, state_tm1 in zip(self.att_mod_list, state_lst_tm1):
                context, state_t = att_mod(query, state_tm1, reuse=reuse)
                context_lst.append(context)
                state_lst_t.append(state_t)
            context = tf.concat(context_lst, axis=-1)
            return context, state_lst_t
