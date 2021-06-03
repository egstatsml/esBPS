"""
Implements a few functions that will be used for PDMP methods

#### References
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

import collections


def compute_l2_norm(grads_target_log_prob):
  """ Computes the L2 norm of the gradient of the target

  Function exists to compute norms for the TFP format,
  where all the variables are stored in a list. The built-in
  tf.norm function expects the input to be a single tensor,
  not a list of tensors, so we need to compute it manually.

  Args:
    grads_target_log_prob (list(array)):
      List of arrays for the gradient of each variable.

  Returns:
    L2 norm of the gradient
  """
  # grads_target_log_prob is a list of arrays, so need to compute the
  # norm in multiple steps. Will first element-wise square each array in list
  grads_square = [tf.square(x) for x in grads_target_log_prob]
  # now get the sum over all the variables
  grads_square_sum = sum_list(grads_square)
  # now take sqrt to find the L2 Norm
  grads_norm = tf.math.sqrt(grads_square_sum)
  return grads_square_sum#grads_norm


def compute_dot_prod(input_a, input_b):
  """ Compute dot product between two vectors.

  Args:
    grads_target_log_prob (list(array)):
      List of arrays for the gradient of each variable.
    velocity (list(array)):
      List of arrays for the velocity of each variable.

   Returns:
     dot product of the two
  """
  # to compute the dot product, will perform element-wise
  # multiplication, and then sum everything
  element_wise_prod = [tf.multiply(g, v) for g,v in
                       zip(input_a, input_b)]
  #                     for i in range(len(velocity))]
  #element_wise_prod = [tf.multiply(grads_target_log_prob[i], velocity[i])
  #                     for i in range(len(velocity))]
  # now sum it all together
  dot_prod = sum_list(element_wise_prod)
  return dot_prod


def sum_list(array_list):
  """ Will find the total sum of all array items in a list

  This helper is here because TFP operates over lists of variables,
  and for this kernel we need to find the total sum over all
  elements many times.
  This method basically just calls tf.reduce_sum multiple times:
  once for each array item in the list, and then a second time to sum
  the individual summations of each item found in the previous step.

  Args:
    array_list (list(array)):
      list of arrays which contain the variables we are ineterested in

  Returns:
    Scalar containing the total sum

  Raises:
    ValueError: if input is not a list of tf arrays
  """
  if not mcmc_util.is_list_like(array_list):
    raise ValueError('Expected input arg to be a list of tf variables')
  # find the total sum over each element
  item_sum = [tf.reduce_sum(x) for x in array_list]
  # Now can sum up all the elements above, since each dimension is the same
  # (each element is a scalar, so tensorflow can handle it)
  total_sum = tf.reduce_sum(item_sum)
  return total_sum
