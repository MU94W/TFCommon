import tensorflow as tf
from tensorflow.python.ops import array_ops
from TFCommon.Initializer import gaussian_initializer, random_orthogonal_initializer
from six.moves import xrange

class BahdanauAttentionModule(object):
    """Attention Module
    Args:
        attention_units:    The attention module's capacity (should be proportional to query_units)
        memory:             A tensor, whose shape should be (None, Time, Unit)
        time_major:
    """
    def __init__(self, attention_units, memory, sequence_length=None, time_major=True, mode=0):
        self.attention_units    = attention_units
        self.enc_units          = memory.get_shape()[-1].value

        if time_major:
            memory = tf.transpose(memory, perm=(1,0,2))

        self.enc_length = tf.shape(memory)[1]
        self.batch_size = tf.shape(memory)[0]
        self.mode = mode
        self.mask = array_ops.sequence_mask(sequence_length, self.enc_length, tf.float32) if sequence_length is not None else None

        self.memory = tf.reshape(memory, (tf.shape(memory)[0], self.enc_length, 1, self.enc_units))
        ### pre-compute Uahj to minimize the computational cost
        with tf.variable_scope('attention'):
            Ua = tf.get_variable(name='Ua', shape=(1, 1, self.enc_units, self.attention_units))
        self.hidden_feats = tf.nn.conv2d(self.memory, Ua, [1,1,1,1], "SAME")
    
    def __call__(self, query):

        with tf.variable_scope('attention'):
            # Check if the memory's batch_size is consistent with query's batch_size

            query_units = query.get_shape()[-1].value

            Wa = tf.get_variable(name='Wa', shape=(query_units, self.attention_units))
            Va = tf.get_variable(name='Va', shape=(self.attention_units,),
                                 initializer=tf.constant_initializer(0.0) if self.mode == 0 else tf.constant_initializer(1e-2))
            b  = tf.get_variable(name='b',  shape=(self.attention_units,),
                                 initializer=tf.constant_initializer(0.0) if self.mode == 0 else tf.constant_initializer(0.5))
 
            ### 1st. compute query_feat (query's repsentation in attention module)
            query_feat = tf.reshape(tf.matmul(query, Wa), (-1, 1, 1, self.attention_units))

            ### 2nd. compute the energy for all time steps in encoder (element-wise mul then reduce)
            e = tf.reduce_sum(Va * tf.nn.tanh(self.hidden_feats + query_feat + b), axis=(2,3))

            ### 3rd. compute the score
            if self.mask is not None:
                exp_e = tf.exp(e)
                exp_e = exp_e * self.mask
                alpha = tf.divide(exp_e, tf.reduce_sum(exp_e, axis=-1, keep_dims=True))
            else:
                alpha = tf.nn.softmax(e)

            ### 4th. get the weighted context from memory (element-wise mul then reduce)
            context = tf.reshape(alpha, (tf.shape(query)[0], self.enc_length, 1, 1)) * self.memory
            context = tf.reduce_sum(context, axis=(1,2))

            return context, alpha

class LuongAttentionModule(object):
    """Attention Module
    Args:
        attention_units:    The attention module's capacity (should be proportional to query_units)
        memory:             A tensor, whose shape should be (None, Time, Unit)
        time_major:
    """
    def __init__(self, attention_units, memory, time_major=True):
        self.attention_units    = attention_units
        self.enc_units          = memory.get_shape()[-1].value

        if time_major:
            memory = tf.transpose(memory, perm=(1,0,2))

        self.enc_length = tf.shape(memory)[1]
        self.batch_size = tf.shape(memory)[0]

        self.memory = tf.reshape(memory, (tf.shape(memory)[0], self.enc_length, 1, self.enc_units))
        ### pre-compute Uahj to minimize the computational cost
        with tf.variable_scope('attention'):
            Ua = tf.get_variable(name='Ua', shape=(1, 1, self.enc_units, self.attention_units),
                                 initializer=gaussian_initializer(mean=0.0, std=0.001))
        self.hidden_feats = tf.nn.conv2d(self.memory, Ua, [1,1,1,1], "SAME")
    
    def __call__(self, query):

        with tf.variable_scope('attention'):
            # Check if the memory's batch_size is consistent with query's batch_size

            query_units = query.get_shape()[-1].value

            Wa = tf.get_variable(name='Wa', shape=(query_units, self.attention_units),
                                 initializer=gaussian_initializer(mean=0.0, std=0.001))
            Va = tf.get_variable(name='Va', shape=(self.attention_units,),
                                 initializer=tf.constant_initializer(0.0))
            b  = tf.get_variable(name='b',  shape=(self.attention_units,),
                                 initializer=tf.constant_initializer(0.0))
 
            ### 1st. compute query_feat (query's repsentation in attention module)
            query_feat = tf.reshape(tf.matmul(query, Wa), (-1, 1, 1, self.attention_units))

            ### 2nd. compute the energy for all time steps in encoder (element-wise mul then reduce)
            e = tf.reduce_sum(query_feat * self.hidden_feats, axis=(2,3))

            ### 3rd. compute the score
            alpha = tf.nn.softmax(e)

            ### 4th. get the weighted context from memory (element-wise mul then reduce)
            context = tf.reshape(alpha, (tf.shape(query)[0], self.enc_length, 1, 1)) * self.memory
            context = tf.reduce_sum(context, axis=(1,2))

            return context, alpha

