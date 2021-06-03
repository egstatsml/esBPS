import math
from datetime import datetime
from functools import partial

import os
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

from tbnn.utils import utils, display, summarise

from tbnn.pdmp.model import (plot_density, get_model_state, pred_forward_pass,
                             bnn_log_likelihood, bnn_joint_log_prob_fn,
                             bnn_neg_joint_log_prob_fn, get_map, get_mle,
                             trace_fn, graph_hmc, nest_concat, build_prior,
                             build_network, cov_pbps_main, pbps_main,
                             bps_main, hmc_main, nuts_main, plot_pred_posterior)
import neptune
import sys
import argparse
from absl import logging
tfd = tfp.distributions
from tbnn.embedded_vi.embedded_vi import EmbeddedVIKernel, DenseReparameterizationMAP
from tbnn.vi import utils as vi_utils

import pickle


def get_data(data, split='100_0_0'):
  dataset = utils.load_dataset(data, split)
  return (dataset.x_train, dataset.x_test), (dataset.y_train, dataset.y_test), dataset.dimension_dict


# def examine_rate(model, bnn_neg_joint_log_prob,
#                  state, X_train, y_train, num_samp=1000):
#   kernel = BPSKernel(
#     target_log_prob_fn=bnn_neg_joint_log_prob,
#     store_parameters_in_results=True,
#     lambda_ref=1.5)
#   # run bootstrap to initialise velocity component
#   bps_results = kernel.bootstrap_results(state)
#   velocity = bps_results.velocity
#   # now iterate over the time steps to evaluate the
#   print('velocity = {}'.format(velocity))
#   time_dt = tf.constant(0.01, dtype=tf.float32)
#   time = tf.Variable(0.0, dtype=tf.float32)
#   test = np.zeros(num_samp)
#   for i in range(0, 1000):
#     test[i] = kernel.examine_event_intensity(state, velocity, time).numpy()
#     time = time + time_dt
#   time_arr = np.linspace(0, 0.01 * 1000.0, 1000)
#   plt.figure()
#   plt.plot(time_arr, test)
#   plt.xlabel('time')
#   plt.ylabel('IPP intensity')
#   plt.savefig('ipp_test.eps')


def examine_rate(model, bnn_neg_joint_log_prob,
                 state, X_train, y_train, out_dir, num_samp=1000):
  kernel = BPSKernel(
    target_log_prob_fn=bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=1.0)
  init_state = [tf.convert_to_tensor(x) for x in state]
  for test_iter in range(0, 10):
    print('eval loop {}'.format(test_iter))
    bps_results = tfp.mcmc.sample_chain(
      num_results=10,
      current_state=init_state,
      return_final_kernel_results=True,
      kernel=kernel)
    samples = bps_results.all_states
    # initialise stafe for next iter
    init_state = [x[-1, ...] for x in samples]
    # final kernel results used to initialise next call of loop
    kernel_results = bps_results.final_kernel_results
    velocity = kernel_results.velocity
    # now iterate over the time steps to evaluate the
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
    plt.savefig(os.path.join(out_dir, 'ipp_test_{}.png'.format(test_iter)))
    np.save(os.path.join(out_dir, 'conv_time_array_{}.npy'.format(test_iter)), time_arr)
    np.save(os.path.join(out_dir, 'conv_test_array_{}.npy'.format(test_iter)), test)


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
      p_scale = 1.0#2.0 *tf.sqrt(2.0 / tf.cast(config['dense_param']['dim'], dtype=tf.float32))
    else:
      # an incorrect layer type was supplied, so raise an error
      err_str = 'Incorrect layer supplied here. expected layer of type'\
                'of either `dense` or `conv`, but got {}'.format(config['type'])
      raise ValueError(err_str)
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=1.0))
  return weights_prior, bias_prior


def main(args):
  (X_train, X_test), (y_train, y_test), data_dimension_dict = get_data(args.data)
  print('y_test shape = {}'.format(y_test.shape))
  #model = build_network(layers)
  model = build_network(args.config, X_train, data_dimension_dict)
  weight_prior_fns, bias_prior_fns = build_prior(model.config_data['layers'])
  print('main model likelihood = {}'.format(model.likelihood_fn))
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
                              num_iters=10000, save_every=100)
  #map_initial_state = get_mle(model, X_train, y_train,
  #                            num_iters=10000, save_every=100)
  #map_initial_state = get_model_state(model)
  print('map_initial_state = {}'.format(map_initial_state))
  pred = pred_forward_pass(model,
                           map_initial_state[::2], map_initial_state[1::2],
                           X_train.astype(np.float32))
  tf.print('pred = {}'.format(pred), output_stream=sys.stdout)
  #pred = model(X_train)
  pred = np.round(tf.keras.activations.sigmoid(pred)).astype(np.int)
  plt.scatter(X_train[pred.ravel()==0, 0], X_train[pred.ravel()==0, 1], color='b')
  plt.scatter(X_train[pred.ravel()==1, 0], X_train[pred.ravel()==1, 1], color='r')
  #plt.scatter(X_test, y_test, color='b', alpha=0.5)
  plt.savefig(os.path.join(args.out_dir, 'pred_logistic_map.png'))
  #examine_rate(model, bnn_neg_joint_log_prob, map_initial_state,
  #             X_train, y_train, args.out_dir)
  data_size = X_train.shape[0]
  batch_size = data_size

  if args.bps:
    bps_main(model, args.ipp_sampler, args.num_results,
             args.num_burnin, args.out_dir,
             bnn_neg_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, batch_size, data_size, data_dimension_dict,
             plot_results=~args.no_plot)
  if args.cov_pbps:
    cov_pbps_main(model, args.ipp_sampler, args.num_results,
                  args.num_burnin, args.out_dir,
                  bnn_neg_joint_log_prob, map_initial_state,
                  X_train, y_train, X_test, y_test,batch_size, data_size, data_dimension_dict,
                  plot_results=~args.no_plot,
                  num_steps_between_results=args.num_steps_between_results)
  if args.pbps:
    pbps_main(model, args.ipp_sampler, args.num_results,
              args.num_burnin, args.out_dir,
              bnn_neg_joint_log_prob, map_initial_state,
              X_train, y_train, X_test, y_test, batch_size, data_size, data_dimension_dict,
              plot_results=~args.no_plot,
              num_steps_between_results=args.num_steps_between_results)
  if args.hmc:
    hmc_main(model, args.num_results, args.num_burnin,
             bnn_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, data_dimension_dict)
  if args.nuts:
    nuts_main(model, args.num_results, args.num_burnin, args.out_dir,
              bnn_joint_log_prob, map_initial_state,
              X_train, y_train, X_test, y_test, data_dimension_dict)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_keras',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('config', type=str,
                      help='path to JSON config file')
  parser.add_argument('--out_dir', type=str, default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--data', type=str, default='moons',
                      help='data set to use here')
  parser.add_argument('--bps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run BPS')
  parser.add_argument('--pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run pBPS (from SBPS)')
  parser.add_argument('--cov_pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run Cov precond. BPS')
  parser.add_argument('--hmc', type=bool, default=False, nargs='?',
                      const=True, help='whether to run HMC')
  parser.add_argument('--ipp_sampler', type=str, default='adaptive', nargs='?',
                      help='type of sampling scheme for event IPP')
  parser.add_argument('--nuts', type=bool, default=False, nargs='?',
                      const=True, help='whether to run NUTS')
  parser.add_argument('--num_results', type=int, default=500,
                      help='number of sample results')
  parser.add_argument('--num_burnin', type=int, default=500,
                      help='number of burnin samples')
  parser.add_argument('--num_steps_between_results', type=int, default=0,
                      help='number of samples between chain samples (for thinning)')
  parser.add_argument('--exp_name', type=str, default='test-logistic', nargs='?',
                        help='name of experiment (usually should have to change)')
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
    neptune.create_experiment(name='test_logistic',
                              params=exp_params,
                              tags=exp_tags,
                              upload_source_files=[args.config])
  main(args)
