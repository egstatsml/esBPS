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
from tbnn.pdmp.poisson_process import AdaptiveSBPSampler

from tbnn.pdmp.model import (plot_density, get_model_state, pred_forward_pass,
                             bnn_neg_joint_log_prob_fn,
                             bnn_log_likelihood,
                             get_map, get_mle,
                             trace_fn, graph_hmc, nest_concat, set_model_params,
                             build_network,
                             bps_main, hmc_main, nuts_main, pbps_main,
                             get_map_iter)
#import some helper functions
from tbnn.utils import utils, display, summarise
from sklearn.model_selection import train_test_split

import argparse
import os
import sys
import neptune

tfd = tfp.distributions
from tbnn.embedded_vi.embedded_vi import EmbeddedVIKernel, DenseReparameterizationMAP
from tbnn.vi import utils as vi_utils


def get_data(data, split='70_15_15', num_data=1000, num_test=100, random_state=0):
  dataset = utils.load_dataset(data, split)
  # won't use all of the data for now, just using a subset to start off with
  # will just get a subset from the training data
  # X_train, X_test, y_train, y_test = train_test_split(dataset.x_train.astype(np.float32),
  #                                                     dataset.y_train.astype(np.float32),
  #                                                     train_size=num_data,
  #                                                     test_size=num_test,
  #                                                     random_state=random_state)
  # return (X_train, X_test), (y_train, y_test), dataset.dimension_dict
  #return (dataset.x_train, dataset.x_test), (2.0 *(dataset.y_train - 0.5), 2.0 * (dataset.y_test - 0.5)), dataset.dimension_dict
  return dataset.x_train, dataset.x_test, dataset.y_train , dataset.y_test, dataset.dimension_dict


def get_model_state(model):
  """generate starting point for creating Markov chain
        of weights and biases for fully connected NN
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
  """
  return model.trainable_variables


def set_model_params(model, weights_list):
  model.layers[0].kernel = weights_list[0]
  return model


def pred_forward_pass(model, weights_list, x):
  model = set_model_params(model, weights_list, biases_lists)
  out = model(x)
  out_np = out.numpy()
  return out_np


def logistic_regression_log_likelihood(model):
  def log_likelihood_eval(x):
    pred = model(x)
    log_likelihood_fn = tfd.ProbitBernoulli(probit=pred)
    #log_likelihood_fn = model.likelihood_fn(pred)
    return log_likelihood_fn
  return log_likelihood_eval


def logistic_regression_joint_log_prob_fn(model, weight_prior_fns, X, y):
  def _fn(*args):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      print(args)
      weights_list = args
      # adding prior component from the joint dist.
      lp = sum(
        [tf.reduce_sum(fn.log_prob(w)) for fn, w in zip(weight_prior_fns, weights_list)])
      m = set_model_params(model, weights_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m)
      print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      print('log likelihood shape = {}'.format(log_likelihood_dist.log_prob(y).shape))
      # add the log likelihood now
      lp += tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp
  return _fn


def logistic_regression_neg_joint_log_prob_fn(model, weight_prior_fns, X, y):
  joint_log_prob = logistic_regression_joint_log_prob_fn(model, weight_prior_fns, X, y)
  return lambda *x: -1.0 * joint_log_prob(*x)


def build_prior(layers_config):
  weights_prior = []
  bias_prior = []
  #weights_prior.append(tfd.Normal(loc=0., scale=1.0))
  #bias_prior.append(tfd.Normal(loc=0., scale=1.0))
  for config in layers_config:
    if(config['type'] == 'pool') or (config['type'] == 'flatten'):
      continue
    elif (config['type'] == 'conv'):
      p_scale = 1.0
      #p_scale = tf.sqrt(1.0 / tf.cast(config['conv_param']['num_output'], dtype=tf.float32))
    elif (config['type'] == 'dense'):
      p_scale = 1.0#tf.sqrt(5.0 / tf.cast(config['dense_param']['dim'], dtype=tf.float32))
    else:
      # an incorrect layer type was supplied, so raise an error
      err_str = 'Incorrect layer supplied here. expected layer of type'\
                'of either `dense` or `conv`, but got {}'.format(config['type'])
      raise ValueError(err_str)
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=1.0))
  return weights_prior, bias_prior


def examine_rate(model, bnn_neg_joint_log_prob,
                 state, X_train, y_train, out_dir, num_samp=1000):
  kernel = BPSKernel(
    ipp_sampler=AdaptiveSBPSampler,
    target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=0.1)
  kernel_previous_results = kernel.bootstrap_results(state)
  init_state = state
  for test_iter in range(0, 20):
    bps_results = graph_hmc(
      num_results=200,
      current_state=init_state,
      previous_kernel_results=kernel_previous_results,
      return_final_kernel_results=True,
      kernel=kernel,
      trace_fn=None)
    bps_chain = bps_results.all_states
    # reset the initial state for the next loop
    init_state = [x[-1] for x in bps_chain]
    # final kernel results used to initialise next call of loop
    # and so we can run the examine method
    kernel_previous_results = bps_results.final_kernel_results
    # now getting the velocity and the rate for the current sample
    state = init_state
    velocity = kernel_previous_results.velocity
    time_dt = tf.constant(0.02, dtype=tf.float32)
    time = tf.Variable(0.0, dtype=tf.float32)
    test = np.zeros(num_samp)
    for i in range(0, num_samp):
      test[i] = kernel.examine_event_intensity(state, velocity, time).numpy()
      time = time + time_dt
    time_arr = np.linspace(0, time_dt.numpy() * num_samp, num_samp)
    plt.figure()
    plt.plot(time_arr, test)
    plt.xlabel('time')
    plt.ylabel('IPP intensity')
    plt.savefig(os.path.join(out_dir, 'regression_ipp_test_{}.png'.format(test_iter)))
    plt.savefig(os.path.join(out_dir, 'regression_ipp_test_{}.pdf'.format(test_iter)))
    np.save(os.path.join(out_dir, 'time_array_{}.npy'.format(test_iter)), time_arr)
    np.save(os.path.join(out_dir, 'test_array_{}.npy'.format(test_iter)), test)



def main(args):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
  X_train, X_test, y_train, y_test, data_dimension_dict = get_data(args.data)
  model = build_network(args.config, X_train, data_dimension_dict)
  print(model.layers)
  print(model(X_train).shape)
  model.summary()
  print('model type = {}'.format(model))
  weight_prior_fns, bias_prior_fns = build_prior(model.config_data['layers'])
  bnn_neg_joint_log_prob = bnn_neg_joint_log_prob_fn(model,
                                                     weight_prior_fns,
                                                     bias_prior_fns,
                                                     X_train, y_train)
  bnn_joint_log_prob = bnn_joint_log_prob_fn(model,
                                             weight_prior_fns,
                                             bias_prior_fns,
                                             X_train, y_train)

  # get the initial state for obtaining MAP estimate.
  # This can just be the getting initial values from the model we defined
  initial_state = get_model_state(model)
  print('initial_state = {}'.format(initial_state))
  map_initial_state = get_map(bnn_neg_joint_log_prob, initial_state, model,
                              num_iters=10000, save_every=10000)
  print('map_initial_state = {}'.format(map_initial_state))
  #display._display_accuracy(model, X_train, y_train.reshape(-1, 1), 'Training Data')
  weights_list = map_initial_state[::2]
  biases_list = map_initial_state[1::2]
  pred = pred_forward_pass(model, weights_list, biases_list,
                            X_train.astype(np.float32))
  pred_int = np.round(pred)
  plt.plot(X_train, pred, color='k')
  plt.scatter(X_train, y_train, color='b', alpha=0.25)
  plt.savefig('pred_map.png')
  fig, ax = plt.subplots()
  ax.scatter(X_train[pred_int==0, 0],
             X_train[pred_int==0, 1], color='b')
  ax.scatter(X_train[pred_int==1, 0],
             X_train[pred_int==1, 1], color='r')
  plt.savefig(os.path.join(args.out_dir, 'pred_map.png'))
  examine_rate(model, bnn_neg_joint_log_prob, map_initial_state,
               X_train, y_train, args.out_dir)
  data_size = X_train.shape[0]
  if args.bps:
    bps_main(model, args.ipp_sampler, args.ref, args.num_results,
             args.num_burnin, args.out_dir,
             bnn_neg_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, data_size, data_size, data_dimension_dict,
             plot_results=~args.no_plot)
  if args.cov_pbps:
    cov_pbps_main(model, args.ipp_sampler, args.ref, args.num_results,
                  args.num_burnin, args.out_dir,
                  bnn_neg_joint_log_prob, map_initial_state,
                  X_train, y_train, X_test, y_test,data_size, data_size, data_dimension_dict,
                  plot_results=~args.no_plot,
                  num_steps_between_results=args.num_steps_between_results)
  if args.pbps:
    pbps_main(model, args.ipp_sampler, args.ref, args.num_results,
              args.num_burnin, args.out_dir,
              bnn_neg_joint_log_prob, map_initial_state,
              X_train, y_train, X_test, y_test, data_size, data_size, data_dimension_dict,
              plot_results=~args.no_plot,
              num_steps_between_results=args.num_steps_between_results)
  if args.hmc:
    hmc_main(model, args.num_results, args.num_burnin,
             bnn_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, data_dimension_dict)
  if args.nuts:
    print('here yo')
    nuts_main(model, args.num_results, args.num_burnin, args.out_dir,
              bnn_joint_log_prob, map_initial_state,
              X_train, y_train, X_test, y_test, data_dimension_dict)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_bnn',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('config', type=str,
                      help='path to JSON config file')
  parser.add_argument('--out_dir', type=str, default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of samples per batch')
  parser.add_argument('--data', type=str, default='toy_a',
                      help='data set to use')
  parser.add_argument('--ref', type=float, default=1.0,
                      help='lambda for refresh poisson process')
  parser.add_argument('--bps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run BPS')
  parser.add_argument('--pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run pSBPS for (SBPS paper)')
  parser.add_argument('--cov_pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run Cov precond. BPS')
  parser.add_argument('--ipp_sampler', type=str, default='adaptive', nargs='?',
                      help='type of sampling scheme for event IPP')
  parser.add_argument('--hmc', type=bool, default=False, nargs='?',
                      const=True, help='whether to run HMC')
  parser.add_argument('--nuts', type=bool, default=False, nargs='?',
                      const=True, help='whether to run nuts')
  parser.add_argument('--num_results', type=int, default=100,
                      help='number of sample results')
  parser.add_argument('--num_burnin', type=int, default=100,
                      help='number of burnin samples')
  parser.add_argument('--num_steps_between_results', type=int, default=0,
                      help='number of samples between chain samples (for thinning)')
  description_help_str = ('experiment description'
                          '(place within single quotes \'\'')
  parser.add_argument('--description', type=str, default='test-logistic',
                      nargs='?', help=description_help_str)
  parser.add_argument('--exp_name', type=str, default='test-bnn-regression', nargs='?',
                        help='name of experiment (usually don\'t have to change)')
  parser.add_argument('--no_log', type=bool, default=False,
                      nargs='?', const=True,
                      help='whether should skip logging to neptune or not')
  parser.add_argument('--no_plot', type=bool, default=False,
                      nargs='?', const=True,
                      help='whether should skip plotting or getting pred metrics')

  args = parser.parse_args(sys.argv[1:])
  # if args.bps == False and args.hmc == False and args.nuts == False:
  #   raise ValueError('Either arg for BPS, HMC or NUTS must be supplied')
  # if we are logging info to neptune, initialise the logger now
  exp_params = vars(args)
  exp_tags = [key for key, x in exp_params.items() if (isinstance(x, bool) and (x == True))]
  if(not args.no_log):
    print('logging to neptune')
    neptune.init('ethangoan/{}'.format(args.exp_name))
    neptune.create_experiment(name='test-bnn-regression',
                              params=exp_params,
                              tags=exp_tags,
                              upload_source_files=[args.config,
                                                   './test_bnn_regression.py'])
  main(args)


# def main(args):
#   gpus = tf.config.experimental.list_physical_devices('GPU')
#   tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
#   (X_train, X_test), (y_train, y_test), data_dimension_dict = get_data(args.data)
#   print('X_train.shape = {}'.format(X_train.shape))
#   # training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#   # # shuffle the training data set, repeat and batch it
#   # training_data = training_data.shuffle(X_train.shape[0]).batch(args.batch_size)
#   # print('training size = {}'.format(training_data.cardinality().numpy()))
#   # # make it repeat indefinitely
#   # training_data = training_data.repeat()
#   # # now create an iter object of it
#   # training_iter = iter(training_data)
#   # # get single sample to build the model
#   # X_build, _ = training_iter.next()
#   model = build_network(args.config, X_train, data_dimension_dict)
#   print(model.layers)
#   print(model(X_train).shape)
#   model.summary()
#   print('model type = {}'.format(model))
#   weight_prior_fns, _ = build_prior(model.config_data['layers'])
#   # bnn_joint_log_prob = bnn_joint_log_prob_iter_fn(model,
#   #                                                 weight_prior_fns,
#   #                                                 bias_prior_fns,
#   #                                                 training_iter)

#   logistic_regression_neg_joint_log_prob = logistic_regression_neg_joint_log_prob_fn(model,
#                                                                                      weight_prior_fns,
#                                                                                      X_train, y_train)
#   # get the initial state for obtaining MAP estimate.
#   # This can just be the getting initial values from the model we defined
#   initial_state = get_model_state(model)
#   print('initial_state = {}'.format(initial_state))
#   map_initial_state = get_map(logistic_regression_neg_joint_log_prob, initial_state, model,
#                               num_iters=1000, save_every=10000)
#   print('map_initial_state = {}'.format(map_initial_state))
#   display._display_accuracy(model, X_train, y_train.reshape(-1, 1), 'Training Data')
#   # now train MCMC method if specified
#   # examine_rate(model, bnn_neg_joint_log_prob, map_initial_state,
#   #              X_train, y_train)
#   examine_rate(model, logistic_regression_neg_joint_log_prob, map_initial_state,
#               X_train, y_train, args.out_dir)

#   # if args.bps:
#   #   bps_main(model, args.ipp_sampler, args.num_results,
#   #            args.num_burnin, args.out_dir,
#   #            logistic_regression_neg_joint_log_prob, map_initial_state,
#   #            X_train, X_test, X_test, y_test, data_dimension_dict,
#   #            plot_results=~args.no_plot)
#   # if args.cov_pbps:
#   #   cov_pbps_main(model, args.ipp_sampler, args.num_results,
#   #                 args.num_burnin, args.out_dir,
#   #                 bnn_neg_joint_log_prob, map_initial_state,
#   #                 X_train, X_test, X_test, y_test, data_dimension_dict,
#   #                 args.batch_size, plot_results=~args.no_plot)
#   # if args.hmc:
#   #   hmc_main(model, args.num_results, args.num_burnin,
#   #            bnn_joint_log_prob, map_initial_state,
#   #            X_train, y_train, X_test, y_test, data_dimension_dict)
#   data_size = 1
#   batch_size = 1
#   if args.bps:
#     bps_main(model, args.ipp_sampler, args.num_results,
#              args.num_burnin, args.out_dir,
#              logistic_regression_neg_joint_log_prob, map_initial_state,
#              X_train, y_train, X_test, y_test, data_size, data_size, data_dimension_dict,
#              plot_results=~args.no_plot)
#   if args.cov_pbps:
#     cov_pbps_main(model, args.ipp_sampler, args.num_results,
#                   args.num_burnin, args.out_dir,
#                   logistic_regression_neg_joint_log_prob, map_initial_state,
#                   X_train, y_train, X_test, y_test,data_size, data_size, data_dimension_dict,
#                   plot_results=~args.no_plot,
#                   num_steps_between_results=args.num_steps_between_results)
#   if args.hmc:
#     hmc_main(model, args.num_results, args.num_burnin,
#              logistic_regression_joint_log_prob, map_initial_state,
#              X_train, y_train, X_test, y_test, data_dimension_dict)
#   if args.nuts:
#     print('here yo')
#     nuts_main(model, args.num_results, args.num_burnin, args.out_dir,
#               logistic_regression_joint_log_prob, map_initial_state,
#               X_train, y_train, X_test, y_test, data_dimension_dict)

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser(prog='test_logistic_regression',
#                                    epilog=main.__doc__,
#                                    formatter_class=
#                                    argparse.RawDescriptionHelpFormatter)
#   parser.add_argument('config', type=str,
#                       help='path to JSON config file')
#   parser.add_argument('--out_dir', type=str, default='./out',
#                       help='out directory where data is saved')
#   parser.add_argument('--batch_size', type=int, default=100,
#                       help='Number of samples per batch')
#   parser.add_argument('--data', type=str, default='mnist_im',
#                       help='data set to use')
#   parser.add_argument('--bps', type=bool, default=False, nargs='?',
#                       const=True, help='whether to run BPS')
#   parser.add_argument('--cov_pbps', type=bool, default=False, nargs='?',
#                       const=True, help='whether to run Cov precond. BPS')
#   parser.add_argument('--ipp_sampler', type=str, default='adaptive', nargs='?',
#                       help='type of sampling scheme for event IPP')
#   parser.add_argument('--hmc', type=bool, default=False, nargs='?',
#                       const=True, help='whether to run HMC')
#   parser.add_argument('--num_results', type=int, default=100,
#                       help='number of sample results')
#   parser.add_argument('--num_burnin', type=int, default=100,
#                       help='number of burnin samples')
#   description_help_str = ('experiment description'
#                           '(place within single quotes \'\'')
#   parser.add_argument('--description', type=str, default='test-logistic',
#                       nargs='?', help=description_help_str)
#   parser.add_argument('--exp_name', type=str, default='test-conv', nargs='?',
#                         help='name of experiment (usually don\'t have to change)')
#   parser.add_argument('--no_log', type=bool, default=False,
#                       nargs='?', const=True,
#                       help='whether should skip logging to neptune or not')
#   parser.add_argument('--no_plot', type=bool, default=False,
#                       nargs='?', const=True,
#                       help='whether should skip plotting or getting pred metrics')

#   args = parser.parse_args(sys.argv[1:])
#   # if args.bps == False and args.hmc == False and args.nuts == False:
#   #   raise ValueError('Either arg for BPS, HMC or NUTS must be supplied')
#   # if we are logging info to neptune, initialise the logger now
#   exp_params = vars(args)
#   exp_tags = [key for key, x in exp_params.items() if (isinstance(x, bool) and (x == True))]
#   if(not args.no_log):
#     print('logging to neptune')
#     neptune.init('ethangoan/{}'.format(args.exp_name))
#     neptune.create_experiment(name='test_logistic_regression',
#                               params=exp_params,
#                               tags=exp_tags,
#                               upload_source_files=[args.config,
#                                                    './test_logistic_regression.py'])
#   main(args)
