import tensorflow as tf, numpy as np
from TFCommon.Initializer import gaussian_initializer, random_orthogonal_initializer
import TFCommon.nest as nest

def as_shape(shape):
  """Converts the given object to a TensorShape."""
  if isinstance(shape, tf.TensorShape):
    return shape
  else:
    return tf.TensorShape(shape)

def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.
  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.
  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.
  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    zeros_flat = [
        tf.zeros(
            tf.stack(_state_size_with_prefix(
                s, prefix=[batch_size])),
            dtype=dtype) for s in state_size_flat
    ]
    for s, z in zip(state_size_flat, zeros_flat):
      z.set_shape(_state_size_with_prefix(s, prefix=[None]))
    zeros = nest.pack_sequence_as(structure=state_size,
                                  flat_sequence=zeros_flat)
  else:
    zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
    zeros = tf.zeros(tf.stack(zeros_size), dtype=dtype)
    zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

  return zeros


class _RNNDecoderCell(object):
  """Abstract object representing an RNN Decoder cell.
  The definition of cell in this package differs from the definition used in the
  literature. In the literature, cell refers to an object with a single scalar
  output. The definition in this package refers to a horizontal array of such
  units.
  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  tuple of integers, then it results in a tuple of `len(state_size)` state
  matrices, each with a column size corresponding to values in `state_size`.
  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by the class `MultiRNNCell`,
  or by calling the `rnn` ops several times. Every `RNNCell` must have the
  properties below and implement `__call__` with the following signature.
  """

  def __call__(self, inputs, state, info, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      info: a peak or a context vec
      scope: VariableScope for the created subgraph; defaults to class name.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      state_size = self.state_size
      return _zero_state_tensors(state_size, batch_size, dtype)

  
class GRUDecoderCell(_RNNDecoderCell):
    """Gated Recurrent Unit Decoder Cell, which recept an additional peak or context tensor."""

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

    def __call__(self, x, h_prev, info, scope=None):
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

            # Check if the info size exist.
            info_size = info.get_shape().with_rank(2)[1]
            if info_size is None:
                raise ValueError("Expecting info_size to be set.")

            ### concat x and info for efficiency.
            concated = tf.concat([x, info], axis=-1)
            concated_size = input_size + info_size

            ### get weights for concated tensor.
            W_shape = (concated_size, self._num_units)
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
            r = tf.sigmoid(tf.matmul(concated, Wr) + tf.matmul(h_prev, Ur) + br)
            z = tf.sigmoid(tf.matmul(concated, Wz) + tf.matmul(h_prev, Uz) + bz)

            ### calculate candidate
            h_slash = tf.tanh(tf.matmul(concated, Wh) + tf.matmul(r * h_prev, Uh) + bh)

            ### final cal
            new_h = (1-z) * h_prev + z * h_slash

            return new_h, tuple([new_h])

class AttentionWrapper(_RNNDecoderCell):
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

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            context, alpha = self._attention_module(h_prev)
            output, new_h = self._cell(x, h_prev, context)
            
            return output, new_h


