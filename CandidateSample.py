import tensorflow as tf

def sampled_softmax_loss(label, logit, projection, num_sampled):
    """
    Args:
        label:
        logit:          unscaled log probabilities
        projection:     (W, b)
        num_sampled:
    """
    local_label = tf.reshape(label, shape=(-1,1))
    local_logit = tf.reshape(logit, shape=(-1, logit.get_shape()[-1].value))
    local_Wt    = tf.transpose(projection[0], perm=(1,0))
    local_b     = projection[1]
    loss_sum    = tf.nn.sampled_softmax_loss(weights=local_Wt, biases=local_b,
                                             labels=local_label,
                                             inputs=local_logit,
                                             num_sampled=num_sampled,
                                             num_classes=local_Wt.get_shape()[0].value)
    loss = tf.divide(tf.reduce_sum(loss_sum), tf.cast(tf.size(local_label), dtype=tf.float32))
    return loss
