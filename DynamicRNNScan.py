import tensorflow as tf

def dynamicRNNScan(cell, inputs, time_major=True, scope=None):
    with tf.variable_scope(scope or tf.get_variable_scope()):
        ### 总是处理TimeMajor类型的数据
        if time_major:
            batch_size = tf.shape(inputs)[1]
            max_time_steps = tf.shape(inputs)[0]
        else:
            batch_size = tf.shape(inputs)[0]
            max_time_steps = tf.shape(inputs)[1]
            inputs = tf.transpose(inputs, perm=(1,0,2))
        units = inputs.get_shape()[-1].value
        inputs = tf.reshape(inputs, shape=(max_time_steps, batch_size, units))

        dtype = inputs.dtype
        output_size = cell.output_size
        state = cell.init_state(batch_size, dtype=dtype)
        input_ta = tf.TensorArray(size=max_time_steps, dtype=dtype)
        input_ta = input_ta.unstack(inputs)
        output_ta = tf.TensorArray(size=max_time_steps, dtype=dtype)

        time = tf.constant(0, dtype=tf.int32)
        cond = lambda time, *_: tf.less(time, max_time_steps)
        def body(time, output_ta, state):
            output, state = cell(input_ta.read(time), state)
            output_ta = output_ta.write(time, output)
            return tf.add(time, 1), output_ta, state
    
        _, final_output_ta, final_state = tf.while_loop(cond, body, [time, output_ta, state])

        final_output = tf.reshape(final_output_ta.stack(), shape=(max_time_steps, batch_size, output_size))
    
        if not time_major:
            final_output = tf.transpose(final_output, perm=(1,0,2))

        return final_output, state

def biDynamicRNNScan(cell_fw, cell_bw, inputs, time_major=True, mode='concat', scope=None):
    with tf.variable_scope(scope or tf.get_variable_scope()):
        ### 总是处理TimeMajor类型的数据
        if time_major:
            batch_size = tf.shape(inputs)[1]
            max_time_steps = tf.shape(inputs)[0]
        else:
            batch_size = tf.shape(inputs)[0]
            max_time_steps = tf.shape(inputs)[1]
            inputs = tf.transpose(inputs, perm=(1,0,2))
        units = inputs.get_shape()[-1].value
        inputs_fw = tf.reshape(inputs, shape=(max_time_steps, batch_size, units))

        with tf.variable_scope('forward'):
            final_output_fw, state_fw = dynamicRNNScan(cell_fw, inputs_fw)

        inputs_bw = tf.reverse(inputs_fw, axis=[0])

        with tf.variable_scope('backward'):
            final_output_bw, state_bw = dynamicRNNScan(cell_bw, inputs_bw)

        if mode == 'concat':
            final_output = tf.concat([final_output_fw, final_output_bw], axis=-1)
            state = tf.concat([state_fw, state_bw], axis=-1)
        elif mode == 'add':
            final_output = tf.add(final_output_fw, final_output_bw)
            state = tf.add(state_fw, state_bw)
        elif mode == 'raw':
            final_output = tuple([final_output_fw, final_output_bw])
            state = tuple([state_fw, state_bw])
        else:
            raise NotImplementedError

        return final_output, state


