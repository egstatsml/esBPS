import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tbnn.pdmp.bps import BPSKernel

import sys
import argparse
tfd = tfp.distributions
from tbnn.embedded_vi.embedded_vi import EmbeddedVIKernel, DenseReparameterizationMAP
from tbnn.vi import utils as vi_utils


class MyDense(keras.layers.Dense):

  def call(self, inputs):
    return self.activation(tf.matmul(inputs, self.kernel) + self.bias)
#tf.enable_v2_behavior()


def plot_density(trace, num_chains, num_samples, figsize):
  """a helper function for plotting traces for
  individual parameters"""
  print(trace.shape)
  # if our variables we are plotting the trace for can be represented as a matrix
  if(len(trace.shape) == 3):
    n_rows = trace.shape[1]
    n_cols = trace.shape[2]
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes = axes.reshape(n_rows, n_cols)
    print(axes.shape)
    print(type(axes))
    for row in range(0, n_rows):
      for col in range(0, n_cols):
      # reshape the trace data to seperate for individual chains
        for chain in range(0, num_chains):
          sns.kdeplot(trace[chain:num_samples * (chain + 1), row, col], ax=axes[row, col])
        #axes[row, col].set_xlabel('w_{}_{}'.format(row, col), fontsize=6)
        axes[row, col].set_axis_off()
  # otherwise the trace variables can be represented as a vector
  else:
    #
    n_rows = trace.shape[1]
    fig, axes = plt.subplots(nrows=n_rows, figsize=figsize)
    for row in range(0, n_rows):
      # reshape the trace data to seperate for individual chains
      for chain in range(0, num_chains):
        sns.kdeplot(trace[chain:num_samples * (chain + 1), row], ax=axes[row])
      axes[row].set_xlabel('w_{}'.format(row), fontsize=6)
      axes[row].set_axis_off()
  return fig,  axes


def build_network(layers, activation=tf.nn.tanh):
  # session = keras.backend.get_session()
  # init = tf.global_variables_initializer()
  # session.run(init)
  #model = keras.Sequential()
  inputs = keras.layers.Input(1)
  x = keras.layers.Dense(layers[0], activation=activation,
              bias_initializer='glorot_uniform')(inputs)
  for units in layers[1:]:
    x = keras.layers.Dense(units, activation=activation,
                           bias_initializer='glorot_uniform')(x)
  # add the final layer
  outputs = keras.layers.Dense(1, bias_initializer='glorot_uniform')(x)
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  loss_object = keras.losses.MeanSquaredError()
  model = keras.models.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer, loss=loss_object)
  # print(model.layers)
  return model


def get_model_state(model):
  """generate starting point for creating Markov chain
        of weights and biases for fully connected NN
    Keyword Arguments:
        layers {tuple} -- number of nodes in each layer of the network
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
  """
  #architecture = []
  # the +1 is here as the first layer is an input layer
  #for layer in model.layers[1:]:
  #  architecture.extend([layer.kernel, layer.bias])
  #print(layer.get_weights())
  return model.trainable_variables


def set_model_params(model, weights_list, biases_list):
  # the -1 is here as the first layer is an input layer
  for i in range(1, len(model.layers)):
    #print('bias before = {}'.format(model.layers[i].bias))
    model.layers[i].kernel = weights_list[i - 1]
    model.layers[i].bias = biases_list[i - 1]
    #print('bias after= {}'.format(model.layers[i].bias))
  return model


def dense(X, W, b, activation):
  return activation(tf.matmul(X, W) + b)

def pred_forward_pass(model, weights_list, biases_lists, x):
  # pass through the first layer (MCMC layer)
  model = set_model_params(model, weights_list, biases_lists)
  for layer in model.layers[1:]:
    x = dense(x, layer.kernel, layer.bias, layer.activation)
  out_np = x.numpy()
  return out_np.ravel()


def bnn_log_likelihood(model, weights_list, biases_list):
  def log_likelihood_eval(x):
    # perform forward pass to get preds
    #model = set_model_params(model, weights_list, biases_lists)
    # for i in range(0, len(weights_list) - 1):
    #   x = dense(x, weights_list[i], biases_list[i], tf.nn.tanh)
    # print('final weights')
    # print(weights_list[-1])
    # pred = dense(x, weights_list[-1], biases_list[-1], lambda y: y)
    pred = model(x)
    log_likelihood_fn = tfd.Normal(loc=pred, scale=0.20)
    print('pred = {}'.format(pred))
    return log_likelihood_fn
  return log_likelihood_eval



def bnn_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):
  mse = tf.keras.losses.MeanSquaredError()
  def _fn(*args):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      weights_list = args[::2]
      biases_list = args[1::2]
      # prior log-prob
      # print('weights_list = {}'.format(weights_list))
      # print('biass_list = {}'.format(biases_list))
      # adding prior component from the joint dist.
      lp = sum(
        [tf.reduce_sum(fn.log_prob(w)) for fn, w in zip(weight_prior_fns, weights_list)]
      )
      lp += sum([tf.reduce_sum(fn.log_prob(b)) for fn, b in zip(bias_prior_fns, biases_list)])
      #print(lp.shape)
      # print('lp = {}'.format(lp))
      # lp = lp * 0.1
      # set the model weights and bias params
      #network = build_network_test(weights_list, biases_list)
      #labels_dist = network(X.astype("float32"))
      #lp += tf.reduce_sum(labels_dist.log_prob(y))

      m = set_model_params(model, weights_list, biases_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m, weights_list, biases_list)
      #print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      #out = model(X)
      #lp += mse(y, out)
      # print('log_l = {}'.format(log_likelihood_dist))
      lp += tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp
  return _fn

def bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):
  joint_log_prob = bnn_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
  return lambda *x: -1.0 * joint_log_prob(*x)

def dense(X, W, b, activation):
  return activation(tf.matmul(X, W) + b)

def build_network_test(weights_list, biases_list, activation=tf.nn.tanh):
  def model(X):
    net = X
    print('here')
    print(X.shape)
    i = 0
    for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):
      print('i = {}'.format(i))
      net = dense(net, weights, biases, activation)
    # final linear layer
    net = tf.matmul(net, weights_list[-1]) + biases_list[-1]
    preds = net[:, 0]
    # preds and std_devs each have size N = X.shape(0) (the number of data samples)
    # and are the model's predictions and (log-sqrt of) learned loss attenuations, resp.
    return tfd.Normal(loc=preds, scale=0.20)

  return model


def get_map(target_log_prob_fn, state, model, num_iters=1000, save_every=100):
  """obtain a MAP estimate"""

  # Set up M-step (gradient descent).
  opt = tf.optimizers.SGD(learning_rate=0.0001)
  state_vars = state#[tf.Variable(s) for s in state]
  # #@tf.function(autograph=False, experimental_compile=True)
  # print('state = {}'.format(state))
  # state_vars = model.trainable_variables#state#get_model_state(model)
  # print('model.trainable_variables')
  # print(model.trainable_variables)
  # #state_vars = [tf.Variable(s) for s in state]
  def map_loss():
    return target_log_prob_fn(*state_vars)



  @tf.function
  def minimize():
    #print('state_vars = {}'.format(state_vars))
    print('bias before = {}'.format(model.layers[-1].bias))
    with tf.GradientTape() as tape:
      loss = map_loss()
      grads = tape.gradient(loss, state_vars)
    print('grads = {}'.format(grads))
    opt.apply_gradients(zip(grads, state_vars))#trainable_variables))
    print('bias after = {}'.format(model.layers[-1].bias))

  #
  # def map_loss():
  #   return target_log_prob_fn(*state_vars)

  # #@tf.function
  # def minimize():
  #   opt.minimize(map_loss, state_vars)

  for _ in range(num_iters):
    minimize()

  # return the state of the model now
  return get_model_state(model)


def get_mle(model, x, y, num_iters=1000, save_every=100):
  """obtain a MAP estimate"""
  #@tf.function
  def _fn():
    model.train_on_batch(x, y)

  for _ in range(0, num_iters):
    _fn()

  return get_model_state(model)


def trace_fn(current_state, results, summary_freq=100):
  #step = results.step
  #with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
  #    for idx, tensor in enumerate(current_state, 1):
  #        count = str(math.ceil(idx / 2))
  #        name = "weights_" if idx % 2 == 0 else "biases_" + count
  #        tf.summary.histogram(name, tensor, step=tf.cast(step, tf.int64))
  return results


@tf.function
def graph_hmc(*args, **kwargs):
  """Compile static graph for tfp.mcmc.sample_chain.
    Since this is bulk of the computation, using @tf.function here
    signifcantly improves performance (empirically about ~5x).
    """
  return tfp.mcmc.sample_chain(*args, **kwargs)


def nest_concat(*args):

  return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *args)

def get_data(num_data=100, num_test=100, random_state=0):
  X_train = np.linspace(0, 2 * np.pi, num_data).astype(np.float32)
  X_test = np.linspace(-np.pi, 3 * np.pi, num_test).astype(np.float32)
  y_train = np.sin(X_train) + 0.2 * np.random.randn(*X_train.shape).astype(np.float32)
  y_test = np.sin(X_test) + 0.2 * np.random.randn(*X_test.shape).astype(np.float32)
  train_sort = np.argsort(X_train)
  test_sort = np.argsort(X_test)
  print(train_sort.shape)
  X_train = X_train[train_sort].reshape(-1, 1)
  y_train = y_train[train_sort].reshape(-1, 1)
  X_test = X_test[test_sort].reshape(-1, 1)
  y_test = y_test[test_sort].reshape(-1, 1)
  print(X_train.shape)

  X_scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
  y_scaler = sklearn.preprocessing.StandardScaler().fit(y_train)
  X_train = X_scaler.transform(X_train)
  X_test = X_scaler.transform(X_test)
  y_train = y_scaler.transform(y_train)
  y_test = y_scaler.transform(y_test)

  return (X_train, X_test), (y_train, y_test), (X_scaler, y_scaler)


def build_prior(layers):
  weights_prior = []
  bias_prior = []
  for units in layers:
    p_scale = tf.sqrt(1.0 / tf.cast(units, dtype=tf.float32))
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=p_scale))
  return weights_prior, bias_prior



def bps_main(model, num_results, num_burnin_steps,
             bnn_neg_joint_log_prob, map_initial_state, X_train,
             y_train, X_test, y_test):
  """main method for running BPS on model"""
  print('running bps')
  kernel = BPSKernel(
    target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=0.5)
  # start sampling
  bps_chain = graph_hmc(
    num_results=num_results + num_burnin_steps,
    current_state=map_initial_state,
    kernel=kernel,
    num_steps_between_results=10,
    trace_fn=None)

  # since we aren't performing any adaptive stuff, will include
  # the full chain
  plot_pred_posterior(model, bps_chain, num_results + num_burnin_steps,
                      X_train, y_train, X_test, y_test, 'bps')


def hmc_main(model, num_results, num_burnin_steps,
             bnn_joint_log_prob, map_initial_state, X_train,
             y_train, X_test, y_test):
  """main method for running BPS on model"""
  print('running HMC')

  # hmc_chain = run_hmc_and_plot(map_initial_state,
  #                              bnn_joint_log_prob, num_results=num_results,
  #                              plot_name='keras_test')
  if(num_burnin_steps is None):
    num_burnin_steps =  num_results // 2
  kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=bnn_joint_log_prob,
    num_leapfrog_steps=5,
    step_size=0.02)
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))
  # start sampling
  hmc_chain = graph_hmc(
    num_results=num_results,
    current_state=map_initial_state,
    kernel=kernel,
    num_steps_between_results=10,
    trace_fn=None)

  plot_pred_posterior(model, hmc_chain, num_results,
                      X_train, y_train, X_test, y_test, 'hmc')




def plot_pred_posterior(model, chain, num_results,
                        X_train, y_train,
                        X_test, y_test, name):
  idx = 0
  num_plot = np.min([num_results, 200])
  pred_array = np.zeros([num_plot, y_test.size])
  plt.figure()
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  print(weights_chain[0].shape)
  for i in range(num_results - pred_array.shape[0], num_results):
    weights_list = [x[i, ...] for x in weights_chain]
    biases_list = [x[i, ...] for x in biases_chain]
    pred_array[idx, :] = pred_forward_pass(model, weights_list, biases_list, X_test)
    plt.plot(X_test, pred_array[idx, :], alpha=0.01, color='k')
    idx +=1
  plt.scatter(X_train, y_train, color='b', alpha=0.1)
  plt.savefig('pred_keras_' + name + '.png')
  w = weights_chain[0][:, 0, 0].numpy()
  az.plot_trace(w.reshape([1, num_results]))
  plt.savefig('trace_keras_' + name + '.png')



def main(args):
  num_results = 500
  num_burnin_steps = 500
  layers = [500, 100]
  weight_prior_fns, bias_prior_fns = build_prior(layers)
  (X_train, X_test), (y_train, y_test), _ = get_data(num_data=100)
  model = build_network(layers)

  bnn_joint_log_prob = bnn_joint_log_prob_fn(model,
                                             weight_prior_fns,
                                             bias_prior_fns,
                                             X_train, y_train)

  bnn_neg_joint_log_prob = bnn_neg_joint_log_prob_fn(model,
                                                     weight_prior_fns,
                                                     bias_prior_fns,
                                                     X_train, y_train)

  # get the initial state for obtaining MAP estimate.
  # This can just be the getting initial values from the model we defined
  initial_state = get_model_state(model)
  print('initial_state = {}'.format(initial_state))
  #print(model.layers[0].get_weights())
  #print('initial_state = {}'.format(initial_state))
  map_initial_state = get_map(bnn_neg_joint_log_prob, initial_state, model,
                              num_iters=20000, save_every=100)
  #map_initial_state = get_mle(model, X_train, y_train,
  #                            num_iters=10000, save_every=100)
  #map_initial_state = get_model_state(model)
  print('map_initial_state = {}'.format(map_initial_state))
  pred = pred_forward_pass(model,
                           map_initial_state[::2], map_initial_state[1::2],
                           X_train.astype(np.float32))
  pred = model(X_train)
  plt.plot(X_train, pred, color='k')
  plt.scatter(X_test, y_test, color='b', alpha=0.5)
  plt.savefig('pred_keras_map.png')
  if args.bps:
    bps_main(model, num_results, num_burnin_steps,
             bnn_neg_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test)
  if args.hmc:
    hmc_main(model, num_results, num_burnin_steps,
             bnn_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_keras',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--bps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run BPS')
  parser.add_argument('--hmc', type=bool, default=False, nargs='?',
                      const=True, help='whether to run HMC')
  args = parser.parse_args(sys.argv[1:])
  if args.bps == False and args.hmc == False:
    raise ValueError('Either arg for BPS or HMC must be supplied')
  main(args)
