import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import arviz as az

tfd = tfp.distributions
from tbnn.pdmp.bps import BPSKernel, CovPBPSKernel, PBPSKernel
from tbnn.pdmp.poisson_process import AdaptivePSBPSampler
tf.enable_v2_behavior()


def dense(X, W, b, activation):
  return activation(tf.matmul(X, W) + b)

def build_network(weights_list, biases_list, activation=tf.nn.tanh):
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


def network_forward(X, weights_list, biases_list, activation=tf.nn.tanh):
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
  return preds


def get_initial_state(weight_prior_fns, bias_prior_fns, num_features=1, num_hidden=20, layers=None):
  """generate starting point for creating Markov chain
        of weights and biases for fully connected NN
    Keyword Arguments:
        layers {tuple} -- number of nodes in each layer of the network
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
    """
  if layers is not None:
    assert layers[-1] == 1
  if layers is None:
    layers = (
      num_features,
      num_hidden,
      num_hidden // 2,
      1,
    )
  print('layers = {}'.format(layers))
  architecture = []
  for idx in range(len(layers) - 1):
    print(idx)
    weigths = weight_prior_fns[idx].sample((layers[idx], layers[idx + 1]))
    biases = bias_prior_fns[idx].sample((layers[idx + 1]))
    # weigths = tf.zeros((layers[idx], layers[idx + 1]))
    # biases = tf.zeros((layers[idx + 1]))
    architecture.extend((weigths, biases))
  return architecture


def bnn_joint_log_prob_fn(weight_prior_fns, bias_prior_fns, X, y, *args):
  weights_list = args[::2]
  biases_list = args[1::2]

  # prior log-prob
  lp = sum(
    [tf.reduce_sum(fn.log_prob(w)) for fn, w in zip(weight_prior_fns, weights_list)]
  )
  lp += sum([tf.reduce_sum(fn.log_prob(b)) for fn, b in zip(bias_prior_fns, biases_list)])
  #lp = lp * 0.1
  # likelihood of predicted labels
  network = build_network(weights_list, biases_list)
  labels_dist = network(X.astype("float32"))
  lp += tf.reduce_sum(labels_dist.log_prob(y))
  return lp

def bnn_neg_joint_log_prob_fn(weight_prior, bias_prior, X, y, *args):
  lp =  bnn_joint_log_prob_fn(weight_prior, bias_prior, X, y, *args)
  return -1.0 * lp

def bnn_likelihood_log_prob_fn(X, y, *args):
  weights_list = args[::2]
  biases_list = args[1::2]
  # likelihood of predicted labels
  network = build_network(weights_list, biases_list)
  labels_dist = network(X.astype("float32"))
  lp = tf.reduce_sum(labels_dist.log_prob(y))
  return lp


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


def run_hmc(
  target_log_prob_fn,
  step_size=0.01,
  num_leapfrog_steps=3,
  num_burnin_steps=1000,
  num_adaptation_steps=800,
  num_results=1000,
  num_steps_between_results=0,
  current_state=None,
  logdir="/tmp/data/output/hmc/",
  resume=None):
  """Populates a Markov chain by performing `num_results` gradient-informed steps with a
  Hamiltonian Monte Carlo transition kernel to produce a Metropolis proposal. Either
  that or the previous state is appended to the chain at each step.

  Arguments:
    target_log_prob_fn {callable} -- Determines the HMC transition kernel
    and thereby the stationary distribution that the Markov chain will approximate.

  Returns:
    (chain(s), trace, final_kernel_result) -- The Markov chain(s), the trace created by `trace_fn`
    and the kernel results of the last step.
  """
  assert (current_state, resume) != (None, None)

  # Set up logging.
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = logdir + stamp
  summary_writer = tf.summary.create_file_writer(logdir)

  kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps
  )
  kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    kernel, num_adaptation_steps=num_adaptation_steps
  )
  #kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(target_log_prob_fn=target_log_prob_fn, step_size=0.01, volatility_fn = lambda *args: 0.)
  if resume is None:
    prev_kernel_results = kernel.bootstrap_results(current_state)
    step = 0
  else:
    prev_chain, prev_trace, prev_kernel_results = resume
    step = len(prev_chain)
    current_state = tf.nest.map_structure(lambda chain: chain[-1], prev_chain)

  tf.summary.trace_on(graph=True, profiler=True)
  with summary_writer.as_default():
    tf.summary.trace_export(
      name="mcmc_sample_trace", step=step, profiler_outdir=logdir
    )
    chain, trace, final_kernel_results = graph_hmc(
      kernel=kernel,
      current_state=current_state,
      num_burnin_steps=num_burnin_steps,
      num_results=num_burnin_steps + num_results,
      previous_kernel_results=prev_kernel_results,
      num_steps_between_results=num_steps_between_results,
      trace_fn=partial(trace_fn, summary_freq=20),
      return_final_kernel_results=True,
    )
  summary_writer.close()

  if resume:
    chain = nest_concat(prev_chain, chain)
    trace = nest_concat(prev_trace, trace)

  return chain, trace, final_kernel_results


def run_bps(target_log_prob_fn,
            num_results=1000,
            current_state=None):
  kernel = BPSKernel(
    target_log_prob_fn=target_log_prob_fn,
    store_parameters_in_results=True,
    lambda_ref=1.0)
  # kernel = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
  #   target_log_prob_fn=joint_log_prob,
  #   num_leapfrog_steps=3,
  #   step_size=1.)
  # start sampling
  samples, kernel_results = graph_hmc(
    num_results=num_results,
    current_state=initial_state,
    kernel=kernel)

  return samples, kernel_results, []


def run_bps_test(target_log_prob_fn,
            num_results=1000,
            current_state=None):
  kernel = CovPBPSKernel(
    target_log_prob_fn=target_log_prob_fn,
    store_parameters_in_results=True,
    ipp_sampler=AdaptivePSBPSampler,
    lambda_ref=1.0)
  # kernel = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
  #   target_log_prob_fn=joint_log_prob,
  #   num_leapfrog_steps=3,
  #   step_size=1.)
  # start sampling
  bps_results = graph_hmc(
    num_results=num_results,
    current_state=initial_state,
    return_final_kernel_results=True,
    kernel=kernel)
  samples = bps_results.all_states
  # final kernel results used to initialise next call of loop
  kernel_results = bps_results.final_kernel_results
  # diag_var = [np.var(x, axis=0) for x in samples]
  # kernel_results = kernel_results._replace(preconditioner=diag_var)
  # samples, kernel_results = graph_hmc(
  #   num_results=num_results,
  #   current_state=initial_state,
  #   previous_kernel_results=kernel_results,
  #   kernel=kernel)


  return samples, kernel_results, []


def get_data(num_data=100, test_size=0.1, random_state=0):

  X_train = np.linspace(0, 2 * np.pi, num_data)
  y_train = np.sin(X_train) + 0.2 * np.random.randn(*X_train.shape)
  X_test = np.linspace(-.5, 2 * np.pi + 0.5, num_data)
  y_test = np.sin(X_test) + 0.2 * np.random.randn(*X_test.shape)

  # features_train = np.linspace(0, 2 * np.pi, num_data)
  # labels_train = np.sin(features) + 0.2 * np.random.randn(*features.shape)


  # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  #     features, labels, test_size=test_size, random_state=random_state
  # )
  print(X_train.shape)
  train_sort = np.argsort(X_train)
  test_sort = np.argsort(X_test)
  print(train_sort.shape)
  X_train = X_train[train_sort].reshape(-1, 1)
  y_train = y_train[train_sort].reshape(-1, 1)
  X_test = X_test[test_sort].reshape(-1, 1)
  y_test = y_test[test_sort].reshape(-1, 1)
  # print(X_train.shape)

  X_scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
  y_scaler = sklearn.preprocessing.StandardScaler().fit(y_train)
  X_train = X_scaler.transform(X_train)
  X_test = X_scaler.transform(X_test)
  y_train = y_scaler.transform(y_train)
  y_test = y_scaler.transform(y_test)

  return (X_train, X_test), (y_train, y_test), (X_scaler, y_scaler)

def get_map(target_log_prob_fn, state, num_iters=1000, save_every=100):
  state_vars = [tf.Variable(s) for s in state]
  opt = tf.optimizers.Adam()
  def map_loss():
    return -target_log_prob_fn(*state_vars)

  @tf.function
  def minimize():
    opt.minimize(map_loss, state_vars)

  traces = [[] for _ in range(len(state))]
  for i in range(num_iters):
    if i % save_every == 0:
      for t, s in zip(traces, state_vars):
        t.append(s.numpy())
    minimize()
  return [np.array(t) for t in traces]

def plot_curves(chain, name='plot_curves'):
  weights_list = chain[::2]
  biases_list = chain[1::2]

  train_trace = []
  test_trace = []
  for i in range(len(weights_list[0])):
    network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])(X_train.astype(np.float32))
    train_trace.append(-tf.reduce_mean(network.log_prob(y_train[:, 0])).numpy())
    network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])(X_test.astype(np.float32))
    test_trace.append(-tf.reduce_mean(network.log_prob(y_test[:, 0])).numpy())

  plt.plot(train_trace, label='train')
  plt.plot(test_trace, label='test')
  plt.legend(loc='best')
  plt.savefig(name + '.png')



def run_bps_and_plot(initial_state, num_results=1000, plot_name='bps'):
  chain, trace, final_kernel_results = run_bps_test(
    bnn_neg_joint_log_prob,
    num_results=num_results,
    current_state=initial_state)
  print(chain)

  # print("Acceptance rate:",
  #       trace.inner_results.is_accepted[-1000:].numpy().mean())

  print('ESS full chain')
  for c in chain:
    print("min ESS/step", tf.reduce_min(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000).numpy())
    print("max ESS/step", tf.reduce_max(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000).numpy())
    print("mean ESS/step", tf.reduce_mean(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000).numpy())

  # subsampled = [x[sub_idx, ...] for x in chain]
  weights_list = []
  for w_idx in range(0, len(chain)):
    weight_list = []
    for mcmc_idx in range(0, num_results - 1):
      weights_a = chain[w_idx][mcmc_idx, ...]
      weights_b = chain[w_idx][mcmc_idx + 1, ...]
      weight_list.append((weights_a.numpy() + weights_b.numpy()) / 2.0)
    weights_list.append(np.reshape(np.vstack(weight_list), [-1, *chain[w_idx].shape[1:]]))

  print('ESS subsampled chain')
  sub_idx = np.arange(0, num_results -1, 50)
  subsampled = [x[sub_idx, ...] for x in weights_list]
  for c in subsampled:
    print("sub min ESS/step", tf.reduce_min(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000))
    print("sub max ESS/step", tf.reduce_max(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000))
    print("sub mean ESS/step", tf.reduce_mean(tfp.mcmc.effective_sample_size(c[-1000:, ...]) / 1000))



  return [x.numpy() for x in chain]#subsampled#chain


def run_hmc_and_plot(initial_state, num_results=1000, plot_name='hmc'):
  chain, trace, final_kernel_results = run_hmc(
    bnn_joint_log_prob,
    num_burnin_steps=5000,
    num_leapfrog_steps=10,
    num_adaptation_steps=10000,
    num_results=num_results,
    step_size=1e-4,
    current_state=initial_state)

  print("Acceptance rate:",
        trace.inner_results.is_accepted[-1000:].numpy().mean())

  for c in chain:
    print("ESS/step", tf.reduce_min(tfp.mcmc.effective_sample_size(c[-1000:]) / 1000).numpy())

  for c in chain:
    print(c.shape)
  plt.figure()
  plt.title("Chains")
  for i in range(10):
    plt.plot(chain[4][:, i, 0])
  plt.savefig(plot_name + '_chains.png')

  plt.figure()
  plt.title("Step size")
  plt.plot(trace.inner_results.accepted_results.step_size)
  plt.savefig(plot_name + '_step_size.png')
  return chain


def build_prior(layer_num_units):
  weights_prior = []
  bias_prior = []
  for num_units in layer_num_units:
    p_scale = 0.5 * tf.sqrt(1.0 / tf.cast(num_units, dtype=tf.float32))
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=p_scale))
  return weights_prior, bias_prior


def get_layer_units(num_features=1, num_hidden=200):
  layers = (
      num_features,
      num_hidden,
      num_hidden // 2,
      1,
    )
  return layers



def examine_rate(model, bnn_neg_joint_log_prob,
                 state, X_train, y_train, num_samp=1000):
  kernel = CovPBPSKernel(
    target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=0.0001)
  bps_results = kernel.bootstrap_results(state)
  for test_iter in range(0, 10):
    state, bps_kernel_results = kernel.one_step(state, bps_results)
    velocity = bps_kernel_results.velocity
    # bps_results = tfp.mcmc.sample_chain(num_results=1,
    #                                     current_state=state,
    #                                     kernel=kernel,
    #                                     trace_fn=None)
    print(bps_results)
    velocity = bps_results.velocity
    preconditioner = bps_results.preconditioner
    # run bootstrap to initialise velocity component
    #bps_results = kernel.bootstrap_results(state)
    # now iterate over the time steps to evaluate the
    #print('velocity = {}'.format(velocity))
    time_dt = tf.constant(0.0001, dtype=tf.float32)
    time = tf.Variable(0.0, dtype=tf.float32)
    test = np.zeros(num_samp)
    for i in range(0, num_samp):
      test[i] = kernel.examine_event_intensity(state, velocity, preconditioner, time).numpy()
      time = time + time_dt
    time_arr = np.linspace(0, time_dt.numpy() * num_samp, 1000)
    plt.figure()
    plt.plot(time_arr, test)
    plt.xlabel('time')
    plt.ylabel('IPP intensity')
    plt.savefig('regression_ipp_test_{}.png'.format(test_iter))
    plt.savefig('regression_ipp_test_{}.pdf'.format(test_iter))
    np.save('time_array.npy', time_arr)
    np.save('test_array.npy', test)


if __name__ == '__main__':
  num_results = 20000
  layer_num_units = get_layer_units()
  weight_prior_fns, bias_prior_fns = build_prior(layer_num_units)
  (X_train, X_test), (y_train, y_test), scalers = get_data(num_data=1000)

  bnn_joint_log_prob = partial(
    bnn_joint_log_prob_fn, weight_prior_fns, bias_prior_fns, X_train, y_train[:, 0]
  )

  print('l = {}'.format(layer_num_units))
  initial_state = get_initial_state(weight_prior_fns, bias_prior_fns, layers=layer_num_units)

  bnn_likelihood_log_prob = partial(
    bnn_likelihood_log_prob_fn, X_train, y_train[:, 0]
  )

  bnn_neg_joint_log_prob = partial(
    bnn_neg_joint_log_prob_fn, weight_prior_fns, bias_prior_fns, X_train, y_train[:, 0]
  )


  z = 0
  #print(initial_state)
  for s in initial_state:
    print("State shape", s.shape)
    z += s.shape.num_elements()
  print("Total params", z)

  # run HMC
  # hmc_chain = run_hmc_and_plot(initial_state, 'default_hmc')
  # plot_curves([c[::50] for c in hmc_chain], name='hmc_chains')
  # plt.ylim(-1, 2)
  # plt.yticks(np.linspace(-1, 2, 16));

  # get MAP
  map_trace = get_map(bnn_joint_log_prob, initial_state, num_iters=1000, save_every=100)
  map_initial_state = [tf.constant(t[-1]) for t in map_trace]
  for x in map_initial_state:
    print(x.shape)
  # HMC from MAP
  #hmc_from_map_chain = run_bps_and_plot(map_initial_state, num_results=num_results,
  #                                      plot_name='hmc_from_map')
  weights_list = map_initial_state[::2]
  biases_list = map_initial_state[1::2]
  pred = network_forward(X_train.astype(np.float32), weights_list, biases_list)
  plt.plot(X_train, pred, color='k')
  plt.scatter(X_test, y_test, color='b', alpha=0.5)
  plt.savefig('pred_map.png')
  print(map_initial_state)
  # model = build_network(weights_list, biases_list)
  # examine_rate(model, bnn_neg_joint_log_prob,
  #                map_initial_state, X_train, y_train, num_samp=1000)
  hmc_from_map_chain = run_bps_and_plot(map_initial_state, num_results=num_results,
                                        plot_name='hmc_from_map')
  weights_chain = hmc_from_map_chain[::2]
  biases_chain = hmc_from_map_chain[1::2]
  num_returned_samples = weights_chain[0].shape[0]
  # perform prediction for each iteration
  sample_idx = np.arange(500, num_returned_samples, 10)
  num_plot = sample_idx.size
  pred = np.zeros([num_plot, y_test.size])
  plt.figure()
  pred_idx = 0
  for i in sample_idx:
    weights_list = [x[i, ...] for x in weights_chain]
    biases_list = [x[i, ...] for x in biases_chain]
    pred[pred_idx, :] = network_forward(X_test.astype(np.float32), weights_list, biases_list)
    plt.plot(X_test, pred[pred_idx, :], alpha=0.05, color='k')
    pred_idx += 1

  plt.scatter(X_train, y_train, color='b', alpha=0.01)
  plt.savefig('pred.png')
  plt.savefig('pred.pdf')
  #print(pred)
  print(weights_chain[0].shape)
  # samples = np.array(weights_chain[0]).reshape(weights_chain[0].size, -1).T
  # corr = np.corrcoef(samples)
  # fig, ax = plt.subplots()
  # ax0 = ax.matshow(corr)
  # fig.colorbar(ax0, ax=ax)
  # plt.savefig('corr.pdf')
  #plot_curves([c[::50] for c in hmc_from_map_chain])
  #plt.ylim(-1, 2)
  #plt.yticks(np.linspace(-1, 2, 16));
  for i in range(0, len(weights_chain)):
    print('weight_chain[{}] shape = {}'.format(i, weights_chain[i].shape))
  plt.figure()
  for layer_idx in range(0, len(weights_chain)):
    for param_idx in np.arange(0, weights_chain[layer_idx].shape[1], 10):
      sample = np.reshape(weights_chain[layer_idx][1000:,param_idx, 0], [1, num_returned_samples - 1000])
      sample_az = az.from_tfp(posterior=sample)
      print(sample_az.posterior)
      az.plot_trace(sample_az)
      plt.savefig('./bnn_test_figs/trace_test_{}_{}.png'.format(layer_idx, param_idx))
      plt.clf()
      az.plot_autocorr(sample_az, max_lag=sample.size)
      plt.savefig('./bnn_test_figs/autocorr_test_{}_{}.png'.format(layer_idx, param_idx))
      plt.clf()
