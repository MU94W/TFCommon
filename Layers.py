import tensorflow as tf

class EmbeddingLayer(object):
    """Embedding Layer
    """

    def __init__(self, classes, size, initializer=None, reuse=None):
        """
        Args:
            classes[int]: embedding classes.
            size[int]: embedding units(size).
            initializer:
            reuse:
        """
        self.__classes = classes
        self.__size = size
        self.__initializer = initializer
        self.__reuse = reuse

    @property
    def classes(self):
        return self.__classes
    
    @property
    def size(self):
        return self.__size

    def __call__(self, input_ts, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.__reuse):
            if self.__initializer:
                initializer = self.__initializer
            else:
                # Default initializer for embeddings should have variance=1.
                sqrt3 = math.sqrt(3)    # Uniform(-sqrt(3), sqrt(3)) has variance=1.
                initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            embedding = tf.get_variable(name="embedding", shape=(self.classes, self.size), \
                    initializer=initializer, dtype=input_ts.dtype)
            embedded = tf.nn.embedding_lookup(embedding, input_ts)
        return embedded

