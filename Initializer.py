import tensorflow as tf, numpy as np
from scipy.stats import ortho_group

def gaussian_initializer(mean, std, use_gpu=True):
    return tf.random_normal_initializer(mean=mean, stddev=std)

def random_orthogonal_initializer(use_gpu=True):
    return tf.orthogonal_initializer()
