import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow import keras
from esbps.pdmp.bps import IterBPSKernel

from esbps.pdmp.model import (plot_density, get_model_state, pred_forward_pass,
                             bnn_neg_joint_log_prob_fn,
                             bnn_log_likelihood,
                             get_map, get_mle,
                             trace_fn, graph_hmc, nest_concat, set_model_params,
                             build_network, bnn_neg_joint_log_prob_fn,
                             bps_iter_main, hmc_main, nuts_main, get_map_iter,
                             save_map_weights)
#import some helper functions
from esbps.utils import utils, display, summarise
from sklearn.model_selection import train_test_split

import argparse
import os
import neptune
import sys
import time
import pickle


tfd = tfp.distributions

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
  return (dataset.x_train, dataset.x_test), (dataset.y_train, dataset.y_test), dataset.x_test_orig, dataset.dimension_dict


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


# def bps_main(model, num_results, num_burnin_steps,
#              bnn_neg_joint_log_prob, map_initial_state,
#              X_test, y_test):
#   """main method for running BPS on model"""
#   print('running bps')
#   # bps_chain = run_bps_and_plot(map_initial_state,
#   #                              bnn_neg_joint_log_prob, num_results=num_results,
#   #                              plot_name='keras_test')
#   kernel = BPSKernel(
#     target_log_prob_fn=bnn_neg_joint_log_prob,
#     store_parameters_in_results=True,
#     lambda_ref=1.0)
#   # start sampling
#   bps_chain = graph_hmc(
#     num_results=num_results + num_burnin_steps,
#     current_state=map_initial_state,
#     kernel=kernel,
#     trace_fn=None)
#   # since we aren't performing any adaptive stuff, will include
#   # the full chain
#   params = [x[-1] for x in bps_chain]
#   weights_chain = bps_chain[::2]
#   biases_chain = bps_chain[1::2]
#   model = set_model_params(model, weights_chain, biases_chain)
#   classification = pred_mean(model, bps_chain, X_test, y_test)
#   plot_pred_posterior(model, bps_chain, X_test, y_test,
#                       classification, './mnist_test', [28, 28], 10)


def pred_mean(model, chain, X, y,
              num_samples=100, batch_size=100, num_classes=10):
  """finds the mean in the predictive posterior"""
  print(chain[0].shape)
  num_samples = np.min([num_samples, chain[0].shape[0]])
  pred_eval = np.zeros([num_samples, X.shape[0], num_classes])
  # create an iterator for the dataset
  images = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
  weights_chain = chain[::2]
  biases_chain = chain[1::2]
  print(weights_chain[0].shape)
  num_total_images = X.shape[0]
  # get a set of the images to perform prediction on
  # setting image index lower value to be zero
  image_lower_idx = 0
  for elem in images.as_numpy_iterator():
    print('image lower idx {}'.format(image_lower_idx))
    # now need to create a set of indicies for the
    # images for each batch
    # lower bound on index was set before the start of loop and is updated at
    # the end of each loop. Need to find upper bound, which will
    # be min(lower_bound + batch_size, num_image
    image_upper_idx = np.min([image_lower_idx + batch_size,
                                num_total_images])
    print('image upper idx {}'.format(image_upper_idx))
    # now have our index limits to make a slice for each data point we
    # are looking at in the current batch
    image_idx = np.arange(image_lower_idx, image_upper_idx)
    # now sample over the posterior samples of interest
    for mcmc_idx in range(num_samples - pred_eval.shape[0], num_samples):
      weights_list = [x[mcmc_idx, ...] for x in weights_chain]
      biases_list = [x[mcmc_idx, ...] for x in biases_chain]
      pred_eval[mcmc_idx, image_idx, ...] = pred_forward_pass(model, weights_list,
                                                              biases_list, elem)
    # now update the lower imager index for the next batch of images
    image_lower_idx += batch_size
  # now get the pred mean and use it to classify each sample
  pred_mean = np.mean(pred_eval, axis=0)
  classification = np.argmax(pred_mean, axis=1)
  print('classification shape = {}'.format(classification.shape))
  return classification


def plot_pred_posterior(model, chain, num_results,
                        X_train, y_train, X_test, y_test,
                        save_dir, plt_dims=[28, 28]):
  """Plot misclassified with credible intervals"""
  classification = pred_mean(model, chain, X_test, y_test)
  display._display_accuracy(model, X_test, y_test, 'Testing Data')
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
    pred_eval = np.zeros([n_samples, locs[0].size, num_classes])
    images = X_test[locs[0], ...]
    weights_chain = chain[::2]
    biases_chain = chain[1::2]
    idx = 0
    for i in range(n_samples - pred_eval.shape[0], n_samples):
      weights_list = [x[i, ...] for x in weights_chain]
      biases_list = [x[i, ...] for x in biases_chain]
      pred_eval[idx, ...] = pred_forward_pass(model, weights_list,
                                              biases_list, images)
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
      pred_mean = np.mean(pred_eval[:, im_idx, :], axis=0)
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([pred_mean - cred_ints[0, :],
                            cred_ints[1, :] - pred_mean])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      #now lets plot it and save it
      plt.subplot(2, 1, 1)
      plt.imshow(images[im_idx].reshape(plt_dims), cmap='gray')
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean.size - 1, pred_mean.size),
                   pred_mean.ravel(), yerr=cred_plot, fmt='o')
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

def examine_rate(model, parent_bnn_neg_joint_log_prob,
                 state, X_train, y_train, out_dir, num_samp=100):
  kernel = IterBPSKernel(
    parent_target_log_prob_fn=parent_bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=0.1)
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
    time_dt = tf.constant(0.002, dtype=tf.float32)
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



def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
  def _fn():
    X, y = dataset_iter.next()
    return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
  return _fn

# def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
#   X, y = dataset_iter.next()
#   return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)


def classification_accuracy(model, input_, labels):
  """get classification accuracy
  Args:
    model (keras.Model):
      neural network model to evaluate
   input_ (np.array):
      input data to be passed through the model
    labels (np.array):
      true label data corresponding to the supplied input_
  Returns:
    classification accuracy
  """
  classification_array, correct_labels = classify(model, input_, labels)
  # now find the accuracy
  accuracy = accuracy_score(correct_labels,
                            classification_array,
                            normalize=True)
  return accuracy


def classify(model, input_, labels):
  """Helper func. classify all the examples
  Args:
  input_ (np.array):
    input data to evaluate
  labels (np.array):
    true labels corresponding to the inputs
  Returns:
  classification_array (np.array):
    list of what the classifier actually predicted
  correct_prediction_eval (np.array):
    list of what the correct prediction labels should be
  """
  data = tf.data.Dataset.from_tensor_slices(
    (input_, labels))
  # batch testing data based on prediction type and the no. test samples
  # send the 2 times the batch size across
  # if the GPU can handle the original batch size for training,
  # then shouldn't have an issue with double for testing
  data = data.batch(np.int(2 * model.batch_size))
  # forming lists to save the output
  classification_list = []
  label_list = []
  # classify each batch and store the results
  for input_batch, labels_batch in data:
    output_ = model(input_batch)
    #print('output_ = {}'.format(output_))
    classification_list.extend(np.argmax(output_, axis=1))
    label_list.extend(np.argmax(labels_batch, axis=1))
  classification_array = np.array(classification_list)
  correct_prediction = np.array(label_list)
  #Wprint('classification_array = {}'.format(classification_array))
  #print('correct_prediction   = {}'.format(correct_prediction))
  return classification_array, correct_prediction



def main(args):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  print('gpus = {}'.format(gpus))
  #tf.config.experimental.set_virtual_device_configuration(
  #  gpus[0],
  #  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  (X_train, X_test), (y_train, y_test), X_test_orig, data_dimension_dict = get_data(args.data)
  # ind = np.random.choice(np.arange(0, X_test.shape[0]), size=100)
  # X_test = X_test[ind, ...]
  # y_test = y_test[ind, ...]
  # X_test_orig = X_test_orig[ind, ...]

  training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  # shuffle the training data set, repeat and batch it
  training_data = training_data.shuffle(X_train.shape[0]).batch(args.batch_size)
  print('training size = {}'.format(training_data.cardinality().numpy()))
  # make it repeat indefinitely
  training_data = training_data.repeat()
  # now create an iter object of it
  training_iter = iter(training_data)
  # get single sample to build the model
  X_build, _ = training_iter.next()
  model = build_network(args.config, X_build, data_dimension_dict)
  #model = ResNet18(10)
  #_ = model(X_build)
  print('model type = {}'.format(model))
  weight_prior_fns, bias_prior_fns = build_prior(model.config_data['layers'])
  # bnn_joint_log_prob = bnn_joint_log_prob_iter_fn(model,
  #                                                 weight_prior_fns,
  #                                                 bias_prior_fns,
  #                                                 training_iter)
  bnn_neg_joint_log_prob = iter_bnn_neg_joint_log_prob(model,
                                                       weight_prior_fns,
                                                       bias_prior_fns,
                                                       training_iter)
  test_fn = iter_bnn_neg_joint_log_prob(model,
                                        weight_prior_fns,
                                        bias_prior_fns,
                                        training_iter)
  # get the initial state for obtaining MAP estimate.
  # This can just be the getting initial values from the model we defined
  initial_state = get_model_state(model)
  print('initial_state = {}'.format(initial_state))
  if args.map_path == None:
    map_start = time.time()
    map_initial_state = get_map_iter(test_fn, initial_state, model,
                                     num_iters=20000, save_every=10000)
    map_end = time.time()
    print('time to find MAP estimate = {}'.format(map_end - map_start))
    print('map_initial_state = {}'.format(map_initial_state))
    accuracy = classification_accuracy(model, X_test, y_test)
    print('Test accuracy from MAP = {}'.format(accuracy))
    # save the MAP weights
    save_map_weights(map_initial_state, args.out_dir)
  else:
    with open(args.map_path, 'rb') as f:
      map_initial_state = pickle.load(f)

  # examine_rate(model, bnn_neg_joint_log_prob, map_initial_state,
  #              X_train, y_train, args.out_dir)
  # now train MCMC method if specified
  # number of samples available for training
  data_size = X_train.shape[0]
  if args.bps:
    bps_iter_main(model, args.ipp_sampler, args.ref, args.num_results,
                  args.num_burnin, args.out_dir,
                  bnn_neg_joint_log_prob, map_initial_state,
                  X_train, X_test, X_test, y_test, X_test_orig,
                  args.batch_size, data_size, data_dimension_dict,
                  plot_results=~args.no_plot)
  if args.cov_pbps:
    cov_pbps_iter_main(model, args.ipp_sampler, args.ref, args.num_results,
                       args.num_burnin, args.out_dir,
                       bnn_neg_joint_log_prob, map_initial_state,
                       X_train, X_test, X_test, y_test,
                       args.batch_size, data_size, data_dimension_dict,
                       plot_results=~args.no_plot)
  if args.pbps:
    pbps_iter_main(model, args.ipp_sampler, args.ref, args.num_results,
                   args.num_burnin, args.out_dir,
                   bnn_neg_joint_log_prob, map_initial_state,
                   X_train, X_test, X_test, y_test, X_test_orig,
                   args.batch_size, data_size, data_dimension_dict,
                   plot_results=~args.no_plot)
  if args.hmc:
    hmc_main(model, args.num_results, args.num_burnin,
             bnn_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, data_dimension_dict)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_conv',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('config', type=str,
                      help='path to JSON config file')
  parser.add_argument('--out_dir', type=str, default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of samples per batch')
  parser.add_argument('--data', type=str, default='mnist_im',
                      help='data set to use')
  parser.add_argument('--ref', type=float, default=1.0,
                      help='lambda for refresh poisson process')
  parser.add_argument('--map_path', type=str, default=None,
                      help='path to load map weights')
  parser.add_argument('--bps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run BPS')
  parser.add_argument('--cov_pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run Cov precond. BPS')
  parser.add_argument('--pbps', type=bool, default=False, nargs='?',
                      const=True, help='sbps')
  parser.add_argument('--ipp_sampler', type=str, default='adaptive', nargs='?',
                      help='type of sampling scheme for event IPP')
  parser.add_argument('--hmc', type=bool, default=False, nargs='?',
                      const=True, help='whether to run HMC')
  parser.add_argument('--num_results', type=int, default=100,
                      help='number of sample results')
  parser.add_argument('--num_burnin', type=int, default=100,
                      help='number of burnin samples')
  description_help_str = ('experiment description'
                          '(place within single quotes \'\'')
  parser.add_argument('--description', type=str, default='test-logistic',
                      nargs='?', help=description_help_str)
  parser.add_argument('--exp_name', type=str, default='test-conv', nargs='?',
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
    neptune.create_experiment(name='test_conv',
                              params=exp_params,
                              tags=exp_tags,
                              upload_source_files=[args.config, './test_conv.py'])
  main(args)
