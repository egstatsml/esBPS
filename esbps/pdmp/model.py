"""model.py

handles implementation of running different samplers for
the different models.

One main component is how the memory of GPU devices available
is handled. For larger models (or even moderately sized models),
storing all samples on the GPU is not feasible (and not necessary).
It might not even be feasible to store them within CPU RAM, so we
save them to file.

Example: say we need to run 1000 samples for our model, but we
can only fit 100 samples in the GPU. To get the full number of samples,
we run the chain to fill up our GPU memopry, quickly exit it, and then
run the chain again (will do this 10 times).

```python

for i in range(0, 10):
  samples = run_chain(num_samples=100)
  # save the current samples to file
  save_samples(samples, i)
```


##### Examples


Getting Concrete function:

Just use the `get_concrete_function()` method attached to any tf function.
```python
concrete = graph_hmc.get_concrete_function(
  num_results=num_results,
  current_state=map_initial_state,
  kernel=kernel,
  trace_fn=None)
```

Saving model graph:

```python
# create a writer
writer = tf.summary.create_file_writer('./conv_debug')
tf.summary.trace_on(graph=True, profiler=True)
####### CALL tf.function HERE #######
with writer.as_default():
  tf.summary.trace_export(
    name='my_func_trace',
    step=0,
    profiler_outdir=out_dir)
```



"""
import time
from abc import ABCMeta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_probability as tfp
from tensorflow import keras
from esbps.pdmp.bps import BPSKernel, IterBPSKernel
from esbps.pdmp.poisson_process import SBPSampler

import tensorflow as tf
from esbps.nn.mlp import MLP
from esbps.nn.conv import Conv
from esbps.utils import utils, display

import os
import sys
from glob import glob
import json
from sklearn.metrics import accuracy_score

tfd = tfp.distributions


class InvalidMCMCLikelihood(ValueError):
  def __init__(self, arg):
    print('Invalid likelihood function specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
      arg, ['normal', 'bernoulli', 'categorical']))
    ValueError.__init__(self)


class MCMCModelUtils(metaclass=ABCMeta):
  """Mixin class to be used for MCMC models


  This class should only implement a few basic functionalities,
  such as setting joint distributions, setting likelihoods,
  setting parameters and running predictive forward passes.

  This class won't implement a lot of the core functionality,
  as it is inteded as a mixin class to be shared across varying
  model types, such as Dense or Conv nets. These model defs
  should implement a lot of the main low lovel functionality.

  This is an abstract class as it should never be used in isolation,
  only as a mixin class.
  """
  def __init__(self, *args, **kwargs):
    # not initialising anything yet
    pass


class MCMCMLP(MLP):
  def __init__(self, prior, posterior,
               json_path=None, dimension_dict=None, name='MCMC_MLP'):
    # explicitly adding likelihoof fn needed for TFP
    super().__init__(prior, posterior, json_path, dimension_dict, name)


  def _get_likelihood(self, arg):
    '''get likelihood for the current model

    For the MCMC models it should be the name of a TFP distribution.
    This method checks that the likelihood supplied is valid, and then
    sets the self.likelihood_fn attribute to the corresponding dist.

    If the supplied likelihood is not valid, than raise an error

    Args:
      args (str):
        name of likelihood fn supplied in the config file

    Returns:
      the input argument if it is of a valid form

    Raises:
      InvalidMCMCLikelihood() if incorrect likelihood supplied
    '''
    arg = arg.lower()
    if(arg == 'normal'):
      self.likelihood_fn = tfd.Normal
    elif(arg == 'bernoulli'):
      self.likelihood_fn = tfd.Bernoulli
    elif(arg == 'categorical'):
      self.likelihood_fn = tfd.OneHotCategorical
    else:
      raise InvalidMCMCLikelihood(arg)
    return arg


class MCMCConv(Conv):
  def __init__(self, prior, posterior,
               json_path=None, dimension_dict=None, name='MCMC_MLP'):
    # explicitly adding likelihoof fn needed for TFP
    super().__init__(prior, posterior, json_path, dimension_dict, name)

  def _get_likelihood(self, arg):
    '''get likelihood for the current model

    For the MCMC models it should be the name of a TFP distribution.
    This method checks that the likelihood supplied is valid, and then
    sets the self.likelihood_fn attribute to the corresponding dist.

    If the supplied likelihood is not valid, than raise an error

    Args:
      args (str):
        name of likelihood fn supplied in the config file

    Returns:
      the input argument if it is of a valid form

    Raises:
      InvalidMCMCLikelihood() if incorrect likelihood supplied
    '''
    arg = arg.lower()
    if(arg == 'normal'):
      self.likelihood_fn = tfd.Normal
    elif(arg == 'bernoulli'):
      self.likelihood_fn = tfd.Bernoulli
    elif(arg == 'categorical'):
      self.likelihood_fn = tfd.OneHotCategorical
    else:
      raise InvalidMCMCLikelihood(arg)
    return arg



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

def get_model_state(model):
  """generate starting point for creating Markov chain
        of weights and biases for fully connected NN
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
  """
  return model.trainable_variables


def set_model_params(model, weights_list, biases_list):
  param_idx = 0
  #print('length of weights and biases = {}, {}'.format(len(weights_list), len(biases_list)))
  #print('weights_list = {}'.format(weights_list))
  #print('number of layers = {}'.format(len(model.layers)))
  for i in range(0, len(model.layers)):
    # set model params, but make sure this current layer isn't a flatten or
    # pooling layer with no params to set
    if(isinstance(model.layers[i], keras.layers.Flatten) or
       isinstance(model.layers[i], keras.layers.MaxPool2D)):
      #print('no trainable params in layer {}'.format(i))
      print(type(model.layers[i]))
      print(model.layers[i])
    else:
      print('i = {}, param_idx = {}'.format(i, param_idx))
      #print('bias before = {}'.format(model.layers[i].bias))
      model.layers[i].kernel = weights_list[param_idx]
      model.layers[i].bias = biases_list[param_idx]
      param_idx += 1
    #print('bias after= {}'.format(model.layers[i].bias))
  return model


@tf.function
def pred_forward_pass(model, weights_list, biases_list, x):
  model = set_model_params(model, weights_list, biases_list)
  weights_shapes = [x.shape for x in weights_list]
  biases_shapes = [x.shape for x in biases_list]
  print('weights shape = {}, bias shape = {}, data shape = {}'.format(weights_shapes, biases_shapes, x.shape))#, output_stream=sys.stdout)
  out = model(x)
  return out


def bnn_log_likelihood(model):
  def log_likelihood_eval(x):
    pred = model(x)
    #print('pred shape = {}'.format(pred.shape))
    #print(model.likelihood_fn)
    #print(model.summary())
    log_likelihood_fn = model.likelihood_fn(pred)
    return log_likelihood_fn
  return log_likelihood_eval


def bnn_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):
  def _fn(*args):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      print(args)
      weights_list = args[::2]
      biases_list = args[1::2]
      # adding prior component from the joint dist.
      lp = sum(
        [tf.reduce_sum(fn.log_prob(w)) for fn, w in zip(weight_prior_fns, weights_list)]
      )
      lp += sum([tf.reduce_sum(fn.log_prob(b)) for fn, b in zip(bias_prior_fns, biases_list)])
      # set the model weights and bias params
      m = set_model_params(model, weights_list, biases_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m)
      print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      #print('X shape = {}'.format(X.shape))
      #print('y shape = {}'.format(y.shape))
      #print('log likelihood dist = {}'.format(log_likelihood_dist))
      #print('log likelihood shape = {}'.format(log_likelihood_dist.log_prob(y).shape))
      # add the log likelihood now
      lp += tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp
  return _fn


def bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):
  joint_log_prob = bnn_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
  return lambda *x: -1.0 * joint_log_prob(*x)


def bnn_joint_log_prob_iter_fn(model, weight_prior_fns, bias_prior_fns, dataset_iter):
  def _fn(*args):
    with tf.name_scope('bnn_joint_log_prob_fn_iter'):
      X, y = dataset_iter.next()
      weights_list = args[::2]
      biases_list = args[1::2]
      # adding prior component from the joint dist.
      lp = sum(
        [tf.reduce_sum(fn.log_prob(w)) for fn, w in zip(weight_prior_fns, weights_list)]
      )
      lp += sum([tf.reduce_sum(fn.log_prob(b)) for fn, b in zip(bias_prior_fns, biases_list)])
      # set the model weights and bias params
      m = set_model_params(model, weights_list, biases_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m)
      print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      # add the log likelihood now
      lp += tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp
  return _fn


def bnn_neg_joint_log_prob_iter_fn(model, weight_prior_fns, bias_prior_fns, dataset_iter):
  joint_log_prob = bnn_joint_log_prob_iter_fn(model, weight_prior_fns, bias_prior_fns, dataset_iter)
  return lambda *x: -1.0 * joint_log_prob(*x)


def get_map(target_log_prob_fn, state, model, num_iters=100000, save_every=100,
            initial_lr=0.01, decay_rate=0.1, decay_steps=10000):
  """obtain a MAP estimate"""
  num_steps = num_iters // decay_steps
  boundaries = np.linspace(decay_steps, num_iters, num_steps)
  values = [initial_lr] + [(initial_lr * decay_rate**i) for i in range(num_steps)]
  print('lr boundaries = {}'.format(boundaries))
  print('lr values = {}'.format(values))
  # Set up M-step (gradient descent).
  learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)#learning_rate)
  state_vars = state#[tf.Variable(s) for s in state]

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

  for _ in range(num_iters):
    minimize()
  keras.backend.clear_session()
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


def get_map_iter(iter_target_log_prob_fn, state, model,
                 num_iters=1000, save_every=100, decay_steps=40000,
                 initial_lr=0.001, decay_rate=0.96):
  """obtain a MAP estimate for method with iteraor over data"""
  # num_steps = num_iters // decay_steps
  # boundaries = np.linspace(decay_steps, num_iters, num_steps)
  # values = [initial_lr] + [(initial_lr * decay_rate**i) for i in range(num_steps)]
  # print('lr boundaries = {}'.format(boundaries))
  # print('lr values = {}'.format(values))
  # # Set up M-step (gradient descent).
  # learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
  #   boundaries, values)
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

  opt = tf.optimizers.Adam(learning_rate=learning_rate)
  #opt = tf.optimizers.Adam(learning_rate=0.001)
  state_vars = state

  def map_loss():
    target_log_prob_fn = iter_target_log_prob_fn()
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

  for _ in range(num_iters):
    minimize()
  keras.backend.clear_session()
  # return the state of the model now
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


def build_prior(layers):
  print('building prior = {}'.format(layers))
  weights_prior = []
  bias_prior = []
  prior_units = [2, *layers]
  for units in prior_units[:-1]:
    p_scale = tf.sqrt(1.0 / tf.cast(units, dtype=tf.float32))
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=p_scale))
  return weights_prior, bias_prior



def find_num_iters_and_loops(num_params, num_samples,
                             gpu_memory=6e9, gpu_ratio=0.05,
                             precision='single'):
  """ Find number of loops needed to get the specified number of samples

  Refer to the docstring of this module for more of a description about why
  this is needed.


  The amount of memory we need is defined by the memory needed for
  all paramaters of a single loop, + the past samples.


  The memory needed for a single sample will be the number of parameters
  times the number of samples desired (plus kernel results from trace_fn).
  Will also need aditional memory loop will be the memory for parameters needed
  for a single iteration, which will be the amount of memory needed for a single
  sample + the extra params (so for BPS will be sample state, gradients and
  velocity, so will be the memory of a single sample time 3).

  Args:
    num_params (int):
      number of WEIGHTS in the model.
    num_samples (int):
      total number of samples needed
    gpu_memory (int):
      GPU memory available in Bytes.
    gpu_ratio (float):
      ratio of memory to allocate for. Should be in (0, 1)
    precision (str):
      if we are using 'single', 'double' or even 'half' floating point precision

  Returns:
    number of iters per loop, and the number of loops needed.
  """
  # find the total amount of memory per weight (in bytes).
  if(precision == 'single'):
    bytes_per_weight = 4
  elif(precision == 'double'):
    bytes_per_weight = 8
  elif(precision == 'half'):
    bytes_per_weight = 2
  else:
    raise ValueError('Incorrect value specified for precision arg.')
  # memory of a single sample will be the number of bytes per weight time
  # the number of samples
  single_sample_memory = np.int(num_params * bytes_per_weight)
  # memory needed for a single iteration is equal to the amount of
  # memory needed for a single sample times three, since
  # need to store the params, velocity and gradients
  single_iter_memory = single_sample_memory * 3
  # now find the total amounbt of memory needed
  total_memory = single_iter_memory + single_sample_memory * num_samples
  print('Total memory = {}'.format(total_memory))
  print('GPU memory = {}'.format(gpu_memory))
  print('float loops = {}'.format(gpu_ratio * gpu_memory / total_memory))
  # now see how many times we will need to loop in order to get this
  # number of samples
  num_loops = np.ceil(total_memory / (gpu_ratio * gpu_memory)).astype(np.int32)
  # now find the number of samples per loop we can get
  num_samples_per_loop = np.floor(num_samples / num_loops).astype(np.int32)
  return num_samples_per_loop, num_loops


def bps_main(model, ipp_sampler_str, lambda_ref, num_results, num_burnin_steps, out_dir,
             bnn_neg_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, batch_size, data_size,
             data_dimension_dict,
             plot_results=True, num_steps_between_results=0):
  """main method for running BPS on model"""
  print('running bps')
  start_time = time.time()
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  ipp_sampler = SBPSampler
  kernel = BPSKernel(
    target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    ipp_sampler=ipp_sampler,
    batch_size=batch_size,
    data_size=data_size,
    lambda_ref=lambda_ref)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # creating the trace function
  trace_fn = lambda _, pkr: pkr.acceptance_ratio
  # BURNIN PHASE
  # start sampling for burnin, and then discard these samples
  bps_chain, _ = graph_hmc(
    num_results=num_burnin_steps,
    current_state=init_state,
    kernel=kernel,
    trace_fn=trace_fn)
  # get the final state of the chain from the previous burnin iter
  init_state = [x[-1] for x in bps_chain]
  end_warmup_time = time.time()
  print('time warmup = {}'.format(end_warmup_time - start_time))
  # SAMPLING PHASE
  # now loop over the actual samples from (hopefully) the posterior
  bps_results, acceptance_ratio = graph_hmc(
    num_results=num_results,
    current_state=init_state,
    num_steps_between_results=num_steps_between_results,
    kernel=kernel,
    trace_fn=trace_fn)
  #    return_final_kernel_results=True,
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(
    np.sum((acceptance_ratio > 1.0))))
  bps_chain = bps_results
  # save these samples to file
  save_chain(bps_chain, out_dir)
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  # plot the results if specified to
  if(plot_results):
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if(y_test.shape[-1] == 1):
      plot_pred_posterior(model, bps_chain, num_results,
                          X_train, y_train, X_test, y_test, out_dir,
                          'bps')
    # otherwise will be classification task, so plot using those methods
    else:
      plot_image_pred_posterior(model, bps_chain, num_results,
                                X_train, y_train, X_test, y_test, 'bps')


def bps_iter_main(model, ipp_sampler_str, lambda_ref, num_results,
                  num_burnin_steps, out_dir, bnn_neg_joint_log_prob,
                  map_initial_state, X_train, y_train, X_test, y_test,
                  X_test_orig, batch_size, data_size, data_dimension_dict,
                  plot_results=True):
  """main method for running BPS on model"""
  print('running bps')
  start_time = time.time()
  # finding the number of samples to perform for each iteration
  ipp_sampler = SBPSampler
  print('map_initial_state bps = {}'.format(map_initial_state))
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  num_samples_per_loop, num_loops = find_num_iters_and_loops(
    num_params, num_results)
  # create the kernel
  kernel = IterBPSKernel(
    parent_target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    ipp_sampler=ipp_sampler,
    batch_size=batch_size,
    data_size=data_size,
    lambda_ref=lambda_ref)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # BURNIN PHASE
  # start sampling for burnin, and then discard these samples
  # we may not be able to fit all the burnin samples in a single loop,
  # so we will loop over a few times if we need to
  print('num_samples_per_loop = {}'.format(num_samples_per_loop))
  print('num_loops = {}'.format(num_loops))
  num_burnin_iters = np.ceil(num_burnin_steps / num_samples_per_loop).astype(np.int)
  # create the trace function
  trace_fn = lambda _, pkr: pkr.acceptance_ratio
  # run bootstrap here to get the initial state for burnin
  # this allows us to start sampling at each loop exactly where we left off
  kernel_previous_results = kernel.bootstrap_results(init_state)
  # now run the burnin phase
  for burnin_iter in range(0, num_burnin_iters):
    print('burnin iter = {}'.format(burnin_iter))
    bps_results = graph_hmc(
      num_results=num_samples_per_loop,
      current_state=init_state,
      kernel=kernel,
      previous_kernel_results=kernel_previous_results,
      return_final_kernel_results=True,
      trace_fn=trace_fn)
    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # get the final state of the chain from the previous burnin iter
    init_state = [x[-1, ...] for x in bps_chain]
  end_warmup_time = time.time()
  print('time warmup = {}'.format(end_warmup_time - start_time))
  # SAMPLING PHASE
  # now loop over the actual samples from (hopefully) the posterior
  acceptance_list = []
  for loop_iter in range(0, num_loops):
    print('loop iter = {}'.format(loop_iter))
    # bps_results = graph_hmc(
    #   num_results=num_samples_per_loop,
    #   current_state=init_state,
    #   previous_kernel_results=kernel_previous_results,
    #   kernel=kernel,
    #   return_final_kernel_results=True,
    #   trace_fn=trace_fn)
    bps_results = graph_hmc(
      num_results=num_samples_per_loop,
      current_state=init_state,
      kernel=kernel,
      previous_kernel_results=kernel_previous_results,
      return_final_kernel_results=True,
      trace_fn=trace_fn)

    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # add the acceptance ratios to the list
    acceptance_list.append(bps_results.trace.numpy())
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # save these samples to file
    save_chain(bps_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
    init_state = [x[-1] for x in bps_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if len(acceptance_list) > 1:
    acceptance_ratio = np.concatenate(acceptance_list)
  else:
    acceptance_ratio = acceptance_list[0]
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(
    np.sum((acceptance_ratio > 1.0))))
  # plot the results if specified to
  if(plot_results):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if(y_test.shape[-1] == 1):
      plot_pred_posterior(model, bps_chain, num_results,
                          X_train, y_train, X_test, y_test, out_dir,
                          'bps')
    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [data_dimension_dict['in_height'],
                    data_dimension_dict['in_width']]
      else:
        plt_dims = [data_dimension_dict['in_height'],
                    data_dimension_dict['in_width'],
                    data_dimension_dict['in_channels']]
      plot_image_iter_pred_posterior(model, out_dir,
                                     X_train, y_train, X_test, y_test,
                                     X_test_orig, plt_dims)


def save_chain(chain, out_dir, loop_iter=0):
  chain_path = os.path.join(out_dir, 'chain_{}.pkl'.format(loop_iter))
  # check the out directory exists, and if it doesn't than make it
  utils.check_or_mkdir(out_dir)
  # now save it
  filehandler = open(chain_path, 'wb')
  pickle.dump(chain, filehandler)



def save_map_weights(map_weights, out_dir):
  map_path = os.path.join(out_dir, 'map_weights.pkl')
  # check the out directory exists, and if it doesn't than make it
  utils.check_or_mkdir(out_dir)
  # now save it
  filehandler = open(map_path, 'wb')
  pickle.dump(map_weights, filehandler)



#@tf.function
def graph_bps(
    num_results,
    num_burnin_steps,
    current_state,
    kernel,
    num_steps_between_results=10,
    trace_fn=None):
  # perform burnin phase
  bps_burnin = tfp.mcmc.sample_chain(
    num_results=1,
    current_state=current_state,
    kernel=kernel,
    num_steps_between_results=num_burnin_steps,
    trace_fn=None)
  # get the final state of the chain from the burnin phase
  keras.backend.clear_session()
  init_state = [x[-1] for x in bps_burnin]
  bps_chain = tfp.mcmc.sample_chain(
    num_results=num_results,
    current_state=init_state,
    kernel=kernel,
    num_steps_between_results=num_steps_between_results,
    trace_fn=None)

  return bps_chain



              # model, args.num_results, args.num_burnin, out_dir,
              # bnn_joint_log_prob, map_initial_state,
              # X_train, y_train, X_test, y_test, data_dimension_dict

def nuts_main(model, num_results, num_burnin_steps, out_dir,
              bnn_joint_log_prob, map_initial_state,
              X_train, y_train, X_test, y_test, data_dimension_dict,
              target_accept_prob=0.95):
  """main method for running HMC on model"""
  print('running NUTS')
  # hmc_chain = run_hmc_and_plot(map_initial_state,
  #                              bnn_joint_log_prob, num_results=num_results,
  #                              plot_name='keras_test')
  if(num_burnin_steps is None):
    num_burnin_steps =  num_results // 2
  kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=bnn_joint_log_prob,
    step_size=0.02)
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # set up kernel to adjust step size
  # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
  #   inner_kernel=kernel,
  #   target_accept_prob=target_accept_prob,
  #   num_adaptation_steps=np.int(num_burnin_steps * 0.8))
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=kernel,
    target_accept_prob=target_accept_prob,
    num_adaptation_steps=np.int(num_burnin_steps * 0.8),
    step_size_getter_fn=lambda pkr: pkr.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size))
  #for i in range(0, 10):
  nuts_chain, divergence = graph_hmc(
    num_results=num_results,
    current_state=init_state,
    kernel=kernel,
    trace_fn=lambda _, pkr: pkr.inner_results.has_divergence)
  print('Number divergences = {}'.format(np.sum(divergence)))
  plot_pred_posterior(model, nuts_chain, num_results,
                      X_train, y_train, X_test, y_test, out_dir,
                      'nuts')
  save_chain(nuts_chain, out_dir, 0)

def hmc_main(model, num_results, num_burnin_steps,
             bnn_joint_log_prob, map_initial_state, X_train,
             y_train, X_test, y_test):
  """main method for running NUTS on model"""
  print('running HMC')
  if(num_burnin_steps is None):
    num_burnin_steps =  num_results // 2
  kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=bnn_joint_log_prob,
    num_leapfrog_steps=1,
    step_size=0.02)
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))
  # start sampling
  for i in range(0, 10):
    hmc_chain = graph_hmc(
      num_results=num_results,
      current_state=map_initial_state,
      kernel=kernel,
      num_steps_between_results=10,
      trace_fn=None)

    plot_pred_posterior(model, hmc_chain, num_results,
                        X_train, y_train, X_test, y_test, 'hmc_{}'.format(i))
    # save this current iter
    print('saving iter {} of total chain'.format(i))
    filehandler = open('hmc_chain_{}'.format(i), 'wb')
    pickle.dump(hmc_chain, filehandler)
    map_initial_state = [t[-1] for t in hmc_chain]


def plot_pred_posterior(model, chain, num_results,
                        X_train, y_train,
                        X_test, y_test, out_dir, name):
  if model.likelihood == 'bernoulli':
    plot_logistic_pred_posterior(model, chain, num_results,
                                 X_train, y_train,
                                 X_test, y_test, out_dir, name)
  elif model.likelihood == 'categorical':
    # get the plot dims from the model
    if(X_test.shape[-1] == 1):
      # plot dimensions will exclude the final channel as a supplied dimension
      # as having a dimension of [height, width, 1] doesn't agree with
      # matplotlibs imshow method
      plt_dims = X_test.shape[1:3]
    else:
      # have the dimensions be that of the full image
      # (so include the channel index, but remove the sample index)
      plt_dims = X_test.shape[1:]
    plot_image_pred_posterior(model, chain, num_results,
                              X_train, y_train,
                              X_test, y_test, name, plt_dims=plt_dims)
  elif model.likelihood == 'normal':
    plot_regression_pred_posterior(model, chain, num_results,
                                   X_train, y_train,
                                   X_test, y_test, out_dir, name)
  else:
    raise ValueError('Incorrect likelihood specified for plotting output')


def plot_regression_pred_posterior(model,
                                   chain, num_results,
                                   X_train, y_train,
                                   X_test, y_test,
                                   out_dir, name):
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  num_returned_samples = weights_chain[0].shape[0]
  # perform prediction for each iteration
  sample_idx = np.arange(0, num_returned_samples - 1, 10)
  num_plot = sample_idx.size
  pred = np.zeros([num_plot, y_test.size])
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  plt.figure()
  plt.scatter(X_train, y_train, color='b', alpha=0.15)
  pred_idx = 0


  for mcmc_idx in sample_idx:
    weights_list_a = [x[mcmc_idx, ...] for x in weights_chain]
    biases_list_a = [x[mcmc_idx, ...] for x in biases_chain]
    weights_list_b = [x[mcmc_idx + 1, ...] for x in weights_chain]
    biases_list_b = [x[mcmc_idx + 1, ...] for x in biases_chain]
    weights_list = [(a + b)/2.0 for a, b in zip(weights_list_a, weights_list_b)]
    biases_list = [(a + b)/2.0 for a, b in zip(biases_list_a, biases_list_b)]
    # weights_list = [x[i, ...] for x in weights_chain]
    # biases_list = [x[i, ...] for x in biases_chain]
    pred[pred_idx, :] = pred_forward_pass(model, weights_list,
                                          biases_list,
                                          X_test.astype(np.float32)).numpy().ravel()
    plt.plot(X_test, pred[pred_idx, :], alpha=0.05, color='k')
    pred_idx += 1
  plt.axis('off')
  print('saving images to {}'.format(out_dir))
  plt.savefig(os.path.join(out_dir, 'pred.png'))
  plt.savefig(os.path.join(out_dir, 'pred.pdf'), bbox_inches='tight')




def plot_logistic_pred_posterior(model,
                                 chain, num_results,
                                 X_train, y_train,
                                 X_test, y_test, out_dir, name):
  idx = 0
  num_plot = num_results#np.min([num_results, 200])
  pred_array = np.zeros([num_plot, y_test.size])
  plt.figure()
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  print(weights_chain[0].shape)
  print(X_test.shape)
  print(y_test.shape)

  for i in range(0, num_results):
    weights_list = [x[i, ...] for x in weights_chain]
    biases_list = [x[i, ...] for x in biases_chain]
    pred_array[idx, :] =  tf.keras.activations.sigmoid(
      pred_forward_pass(model, weights_list, biases_list, X_test).numpy().ravel())
    idx += 1
  pred_mean = np.mean(pred_array, axis=0)
  pred_mean_classification = np.round(pred_mean).astype(np.int)
  print(pred_mean_classification.shape)
  print(X_test.shape)
  plt.scatter(X_test[pred_mean < 0.5, 0],
              X_test[pred_mean < 0.5, 1], color='b')
  plt.scatter(X_test[pred_mean >= 0.5, 0],
              X_test[pred_mean >= 0.5, 1], color='r')
  # plt.scatter(X_test[pred_mean_classification==0, 0],
  #             X_test[pred_mean_classification==0, 1], color='b')
  # plt.scatter(X_test[pred_mean_classification==1, 0],
  #             X_test[pred_mean_classification==1, 1], color='r')
  plt.savefig(os.path.join(out_dir, 'pred_logistic_' + name + '.png'))
  w = weights_chain[0][:, 0, 0].numpy()
  #az.plot_trace(w.reshape([1, num_results]))
  #plt.savefig(os.path.join(out_dir, 'trace_logistic_' + name + '.png'))
  # create a grid to iterate over
  # using method from Thomas Wiecki's blog
  grid = np.mgrid[-1.5:1.5:100j,-1.5:1.5:100j]
  grid_2d = grid.reshape(2, -1).T
  print('grid_2d shape = {}'.format(grid_2d.shape))
  pred_grid = np.zeros([num_results, grid_2d.shape[0]])
  idx = 0
  for i in range(0, num_results):
    weights_list = [x[i, ...] for x in weights_chain]
    biases_list = [x[i, ...] for x in biases_chain]
    pred_grid[idx, :] = tf.keras.activations.sigmoid(
      pred_forward_pass(model, weights_list, biases_list, grid_2d).numpy().ravel())
    idx +=1

  grid_mean = np.mean(pred_grid, axis=0)
  print('grid_mean shape = {}'.format(grid_mean.shape))
  cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0], grid[1], grid_mean.reshape(100, 100), cmap=cmap)
  ax.scatter(X_test[pred_mean_classification==0, 0],
             X_test[pred_mean_classification==0, 1], color='b')
  ax.scatter(X_test[pred_mean_classification==1, 0],
             X_test[pred_mean_classification==1, 1], color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y');
  cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')
  plt.savefig(os.path.join(out_dir, 'grid_mean_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'grid_mean_logistic_' + name + '.pdf'))
  grid_std = np.std(pred_grid, axis=0)
  cmap = sns.cubehelix_palette(light=1, as_cmap=True)
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0], grid[1], grid_std.reshape(100, 100), cmap=cmap)
  ax.scatter(X_test[pred_mean_classification==0, 0],
             X_test[pred_mean_classification==0, 1], color='b')
  ax.scatter(X_test[pred_mean_classification==1, 0],
             X_test[pred_mean_classification==1, 1], color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y');
  cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
  plt.savefig(os.path.join(out_dir, 'grid_var_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'grid_var_logistic_' + name + '.pdf'))

  weights_mean = [np.mean(x, axis=0) for x in weights_chain]
  bias_mean = [np.mean(x, axis=0) for x in biases_chain]
  pred_grid = tf.keras.activations.sigmoid(
    pred_forward_pass(model, weights_list, biases_list, grid_2d).numpy().ravel())
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0], grid[1], grid_mean.reshape(100, 100), cmap=cmap)
  ax.scatter(X_test[pred_mean_classification==0, 0],
             X_test[pred_mean_classification==0, 1], color='b')
  ax.scatter(X_test[pred_mean_classification==1, 0],
             X_test[pred_mean_classification==1, 1], color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y');
  cbar.ax.set_ylabel('MC Approx. p(y=1|x)')
  plt.savefig(os.path.join(out_dir, 'mc_approx_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'mc_approx_logistic_' + name + '.pdf'))

  fig, ax = plt.subplots()
  contour = ax.contour(grid[0], grid[1], grid_mean.reshape(100, 100), levels=6)
  ax.scatter(X_test[pred_mean_classification==0, 0],
             X_test[pred_mean_classification==0, 1], color='b')
  ax.scatter(X_test[pred_mean_classification==1, 0],
             X_test[pred_mean_classification==1, 1], color='r')
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y');
  cbar.ax.set_ylabel('MC Approx. p(y=1|x)')
  plt.savefig(os.path.join(out_dir, 'mc_approx_contour_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'mc_approx_contour_logistic_' + name + '.pdf'))
  print('pred_mean = {}'.format(pred_mean))


def pred_mean(model, chain, X, y,
              num_samples=100, batch_size=100,
              num_classes=10, final_activation=tf.keras.activations.softmax):
  """finds the mean in the predictive posterior"""
  pred_eval_array = pred_eval_fn(model, chain, X, y, num_samples,
                              batch_size, num_classes, final_activation)
  pred_mean = np.mean(pred_eval_array, axis=0)
  classification = np.argmax(pred_mean, axis=1)
  print('classification shape = {}'.format(classification.shape))
  return classification



def pred_eval_fn(model, chain, X, y,
                 num_samples=100, batch_size=100,
                 num_classes=10, final_activation=tf.keras.activations.softmax):
  """finds the mean in the predictive posterior"""
  print(model.summary())
  num_samples = np.min([num_samples, chain[0].shape[0]])
  pred_eval_array = np.zeros([num_samples, X.shape[0], num_classes])
  # create an iterator for the dataset
  images = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  num_total_images = X.shape[0]
  # get a set of the images to perform prediction on
  # setting image index lower value to be zero
  image_lower_idx = 0
  for elem in images.as_numpy_iterator():
    # now need to create a set of indicies for the
    # images for each batch
    # lower bound on index was set before the start of loop and is updated at
    # the end of each loop. Need to find upper bound, which will
    # be min(lower_bound + batch_size, num_image
    image_upper_idx = np.min([image_lower_idx + batch_size,
                                num_total_images])
    # now have our index limits to make a slice for each data point we
    # are looking at in the current batch
    image_idx = np.arange(image_lower_idx, image_upper_idx)
    # now sample over the posterior samples of interest
    for mcmc_idx in range(num_samples - pred_eval_array.shape[0] + 1, num_samples - 1):
      weights_list_a = [x[mcmc_idx, ...] for x in weights_chain]
      biases_list_a = [x[mcmc_idx, ...] for x in biases_chain]
      weights_list_b = [x[mcmc_idx + 1, ...] for x in weights_chain]
      biases_list_b = [x[mcmc_idx + 1, ...] for x in biases_chain]
      weights_list = [(a + b)/2.0 for a, b in zip(weights_list_a, weights_list_b)]
      #print('weights_list = {}'.format(weights_list))
      biases_list = [(a + b)/2.0 for a, b in zip(biases_list_a, biases_list_b)]
      pred_eval_array[mcmc_idx, image_idx, ...] = final_activation(
        pred_forward_pass(model, weights_list, biases_list, elem)).numpy()
    # now update the lower imager index for the next batch of images
    image_lower_idx += batch_size
  return pred_eval_array



def pred_eval_map_fn(model, chain, X, y,
                     batch_size=100, num_classes=10,
                     final_activation=tf.keras.activations.softmax):
  """finds the mean in the predictive posterior"""
  print(model.summary())
  pred_eval_array = np.zeros([X.shape[0], num_classes])
  # create an iterator for the dataset
  images = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
  weights = chain[::2]
  biases = chain[1::2]
  num_total_images = X.shape[0]
  # get a set of the images to perform prediction on
  # setting image index lower value to be zero
  batch_idx = 0
  for elem in images.as_numpy_iterator():
    start = batch_idx * batch_size
    end = start + batch_size
    pred_eval_array[start:end , ...] = final_activation(
      pred_forward_pass(model, weights, biases, elem)).numpy()
    # now update the batch_idx
    batch_idx += 1
  return pred_eval_array



def plot_image_pred_posterior(model, chain, num_results,
                              X_train, y_train, X_test, y_test,
                              X_test_orig, save_dir, plt_dims=[28, 28]):
  """Plot misclassified with credible intervals"""
  classification = pred_mean(model, chain, X_test, y_test,
                             final_activation=tf.keras.activations.softmax)
  #display._display_accuracy(model, X_test, y_test, 'Testing Data')
  num_classes = 10
  # create a figure
  plt.figure()
  # iterate over all classes
  correct_preds = np.argmax(y_test, axis=1)
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    locs = np.where(np.logical_and(classification != correct_preds,
                                   y_test[:, label_i] == 1))
    pred_eval = np.zeros([num_results, locs[0].size, num_classes])
    images = X_test[locs[0], ...]
    plot_images = X_test_orig[locs[0], ...]
    weights_chain = chain[::2]
    biases_chain = chain[1::2]
    idx = 0
    for i in range(num_results - pred_eval.shape[0], num_results):
      weights_list = [x[i, ...] for x in weights_chain]
      biases_list = [x[i, ...] for x in biases_chain]
      pred_eval[idx, ...] = tf.keras.activations.softmax(
        pred_forward_pass(model, weights_list, biases_list, images))
      idx +=1
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    x_tick = np.arange(0, 10)
    for im_idx in range(0, pred_eval.shape[1]):
      # approximate the mean and credible intervals
      cred_ints = display.mc_credible_interval(
        pred_eval[:, im_idx, :].reshape([-1, num_classes]),
        np.array([0.025, 0.975]))
      pred_mean_array = np.mean(pred_eval[:, im_idx, :], axis=0)
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([pred_mean_array - cred_ints[0, :],
                            cred_ints[1, :] - pred_mean_array])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      #now lets plot it and save it
      cmap = 'gray' if len(plt_dims) == 2 else None
      plt.subplot(2, 1, 1)
      print(plt_dims)
      plt.imshow(plot_images[im_idx].reshape(plt_dims), cmap=cmap)
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean_array.size - 1, pred_mean_array.size),
                   pred_mean_array.ravel(), yerr=cred_plot, fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes),
                 x_tick,
                 size='small',
                 rotation='vertical')
      plt.xlabel("class")
      plt.ylabel("Predicted Probability\nwith 95% CI")
      #plt.savefig(os.path.join(save_dir, str(label_i),
      #                         "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps', bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1


def plot_image_iter_pred_posterior(model, save_dir,
                                   X_train, y_train, X_test, y_test,
                                   X_test_orig, plt_dims=[28, 28],
                                   final_activation=tf.keras.activations.softmax):
  """Plot misclassified with credible intervals

  Handles models that were run in an 'iter' state, as
  all the samples would not fit in memory so need to
  iterate over them in the save directory to compute expectations.
  """
  # get a list of all the chain files
  chain_files = glob(os.path.join(save_dir, 'chain_*.pkl'))
  # create a list to store all samples from predictive posterior
  pred_list = []
  idx = 0
  for chain_file in chain_files:
    print('idx = {}'.format(idx))
    idx += 1
    with open(chain_file, 'rb') as f:
      chain = pickle.load(f)
    pred_list.append(pred_eval_fn(model, chain, X_test, y_test,
                                  final_activation=tf.keras.activations.softmax))
  pred_posterior = np.concatenate(pred_list, axis=0)
  # now perform classification based on the mean
  pred_mean_array = np.mean(pred_posterior, axis=0)
  classification = np.argmax(pred_mean_array, axis=1)
  accuracy = accuracy_score(np.argmax(y_test, axis=1),
                            classification, normalize=True)
  print('Test Accuracy = {}'.format(accuracy))
  num_classes = 10
  # create a figure
  plt.figure()
  # iterate over all classes
  correct_preds = np.argmax(y_test, axis=1)
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    locs = np.where(np.logical_and(classification != correct_preds,
                                   y_test[:, label_i] == 1))
    pred_posterior_misclassified = pred_posterior[:, locs[0], ...]
    pred_mean_misclassified = pred_mean_array[locs[0], ...]
    images = X_test[locs[0], ...]
    plt_images = X_test_orig[locs[0], ...]
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    x_tick = np.arange(0, 10)
    for im_idx in range(0, images.shape[0]):
      # approximate the mean and credible intervals
      cred_ints = display.mc_credible_interval(
        pred_posterior_misclassified[:, im_idx, :].reshape([-1, num_classes]),
        np.array([0.025, 0.975]))
      pred_mean_im = pred_mean_misclassified[im_idx, :]
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([pred_mean_im - cred_ints[0, :],
                            cred_ints[1, :] - pred_mean_im])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      cmap = 'gray' if len(plt_dims) == 2 else None
      #now lets plot it and save it
      plt_image = plt_images[im_idx, ...]
      # if the image data is in integer range, divide by 255 to normalise
      # valid float range of [0.0, 1.0]
      if (np.max(plt_image) > 1.0):
        plt_image = plt_image / 255.0
      plt.subplot(2, 1, 1)
      plt.imshow(plt_image.reshape(plt_dims), cmap=cmap)
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean_im.size - 1, pred_mean_im.size),
                   pred_mean_im.ravel(), yerr=cred_plot, fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes),
                 x_tick,
                 size='small',
                 rotation='vertical')
      plt.xlabel("class")
      plt.ylabel("Predicted Probability\nwith 95% CI")
      #plt.savefig(os.path.join(save_dir, str(label_i),
      #                         "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps', bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1


def create_entropy_hist(model, save_dir,
                        X_train, y_train, X_test, y_test, data_name,
                        plt_dims=[28, 28],
                        final_activation=tf.keras.activations.softmax):
  """Plot entropy for the predicted output

  Handles models that were run in an 'iter' state, as
  all the samples would not fit in memory so need to
  iterate over them in the save directory to compute expectations.
  """
  # get a list of all the chain files
  chain_files = glob(os.path.join(save_dir, 'chain_*.pkl'))
  # create a list to store all samples from predictive posterior
  pred_list = []
  idx = 0
  for chain_file in chain_files:
    print('idx = {}'.format(idx))
    idx += 1
    with open(chain_file, 'rb') as f:
      chain = pickle.load(f)
    pred_list.append(pred_eval_fn(model, chain, X_test, y_test,
                                  final_activation=tf.keras.activations.softmax))
  pred_posterior = np.concatenate(pred_list, axis=0)
  # now perform classification based on the mean
  pred_mean_array = np.mean(pred_posterior, axis=0)
  # now find the entropy value here
  entropy = -np.sum(pred_mean_array * np.log2(pred_mean_array + 1e-7), axis=1)
  np.save(os.path.join(save_dir, '{}_entropy.npy'.format(data_name)), entropy)


def create_entropy_hist_map(model, save_dir,
                            X_train, y_train, X_test, y_test, data_name,
                            plt_dims=[28, 28],
                            final_activation=tf.keras.activations.softmax):
  """Plot entropy for the predicted output for MAP
  """
  # get a list of all the chain files
  map_path = os.path.join(save_dir, 'map_weights.pkl')
  # create a list to store all samples from predictive posterior
  pred_list = []
  with open(map_path, 'rb') as f:
    chain = pickle.load(f)
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  pred_map = pred_eval_map_fn(model, chain, X_test, y_test,
                          final_activation=tf.keras.activations.softmax)
  #pred_forward_pass(model, weights_list, biases_list, elem)).numpy()
  # now find the entropy value here
  entropy = -np.sum(pred_map * np.log2(pred_map + 1e-7), axis=1)
  np.save(os.path.join(save_dir, '{}_entropy_map.npy'.format(data_name)), entropy)


def build_network(json_path, x_train, data_dimension_dict):
  with open(json_path) as file_p:
    data = json.load(file_p)
  if(data['posterior'] == 'dense mcmc'):
    model = MCMCMLP("point", "point", json_path,
                    dimension_dict=data_dimension_dict)
  elif(data['posterior'] == 'conv mcmc'):
    model = MCMCConv("point", "point", json_path,
                     dimension_dict=data_dimension_dict)
  else:
    raise ValueError('Unsuitable model type supplied')
  # running one batch of training just to initialise the model
  _ = model(x_train)
  return model
