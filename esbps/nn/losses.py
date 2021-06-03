"""
Helper functions for different losses.

These are generic loss functions, such as MSE, cross entropy etc.
Aimed for point estimate networks, but may be useful for some
Bayes applications aswell.

These are mostly wrappers so we can write use them similar to
how we handle more comprehensive losses like Bayes by Backprop.

Given this, additional terms that are used by these more comprehensive
methods are wrapped up in the *args term and not actually used.
Am not putting the pylint ignore on these though, in case I ever want
to change it/so I don't forget why I did it.

All these functions have the same arguments and returns

Args:
  model (Neural Network Object):
    Object that holds all the layers and their definitions. Most likely
    will be a point estimate network
  label (tf.Variable):
    Variable that holds the labels for the current batch
  output_ (tf.Variable):
    the output of the network
  *args (list):
    contains extra elements that are typically used by more comprehensive
    loss functions. Include (writer, step index, batch number)
  **kwargs (dict):
    dictionary with extra args not typically used by these loss functions
   Include (num_samples)

Returns:
  cost for current mini-batch
"""

import tensorflow as tf


def mse_loss(model, input_, label, *args, **kwargs):
  """computing the mean squared error"""
  output_ = model(input_)
  cost = tf.losses.mean_squared_error(label, output_)
  return tf.reduce_mean(cost)

def mle_loss(model, input_, label, *args, **kwargs):
  """computing the maximum likelihood estimate by maximising log-likelihood"""
  output_ = model(input_)
  cost = -1.0 * model.log_likelihood(label, output_)
  return tf.reduce_mean(cost)

def map_loss(model, input_, label, *args, **kwargs):
  """computes maximum aposteriori estimate"""
  output_ = model(input_)
  cost = -1.0 * model.log_likelihood(label, output_) + model.regularisation()
  return tf.reduce_mean(cost)
