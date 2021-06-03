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
from tbnn.pdmp.bps import IterBPSKernel

#import some helper functions
from tbnn.utils import utils, display, summarise
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


import argparse
import os
import neptune
import sys
import time
import pickle

from resnet18 import ResNet18


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
  return (dataset.x_train, dataset.x_test), (dataset.y_train, dataset.y_test), dataset.x_test_orig, dataset.dimension_dict



def build_model(model_type, x_train):
  # KL divergence weighted by the number of training samples, using
  # lambda function to pass as input to the kernel_divergence_fn on
  # flipout layers.
  kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(x_train.shape[0], dtype=tf.float32))
  if model_type == 'lenet5':
    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    model = tf.keras.models.Sequential([
      tfp.layers.Convolution2DFlipout(
        6, kernel_size=5, padding='SAME',
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'),
      tfp.layers.Convolution2DFlipout(
        16, kernel_size=5, padding='SAME',
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'),
      tfp.layers.Convolution2DFlipout(
        120, kernel_size=5, padding='SAME',
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(
        84, kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu),
      tfp.layers.DenseFlipout(
        10, kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.softmax)
    ])
  else:
    model = tf.keras.models.Sequential([
      tfp.layers.Convolution2DFlipout(
        16, kernel_size=3,
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.elu),
      tfp.layers.Convolution2DFlipout(
        32, kernel_size=3,
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.elu),
      tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'),
      tfp.layers.Convolution2DFlipout(
        32, kernel_size=3,
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.elu),
      tfp.layers.Convolution2DFlipout(
        43, kernel_size=3,
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.elu),
      tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(
        256, kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.elu),
      tfp.layers.DenseFlipout(
        10, kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.softmax)
    ])
  # Model compilation.
  optimizer = tf.keras.optimizers.Adam(lr=0.001)
  # We use the categorical_crossentropy loss since the MNIST dataset contains
  # ten labels. The Keras API will then automatically add the
  # Kullback-Leibler divergence (contained on the individual layers of
  # the model), to the cross entropy loss, effectively
  # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
  model.compile(optimizer, loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)
  # call the model with a couple batches to build the shape of everything
  model(x_train[0:2, ...])
  return model



def main(args):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  print('gpus = {}'.format(gpus))
  #tf.config.experimental.set_virtual_device_configuration(
  #  gpus[0],
  #  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  (X_train, X_test), (y_train, y_test), X_test_orig, data_dimension_dict = get_data(args.data)
  training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  # shuffle the training data set, repeat and batch it
  training_data = training_data.shuffle(X_train.shape[0]).batch(args.batch_size)
  print('training size = {}'.format(training_data.cardinality().numpy()))
  model = build_model(args.model_type, X_train)
  print(model.summary())
  # now fit the model
  bbb_start = time.time()
  history = model.fit(training_data,
                      epochs=args.epochs)
  bbb_end = time.time()
  print('time to fit BBB = {}'.format(bbb_end - bbb_start))
  # now get predictions
  probs = tf.stack([model.predict(X_test, verbose=1)
                    for _ in range(args.num_monte)], axis=0)
  mean_probs = tf.reduce_mean(probs, axis=0)
  # now get classification
  classification = tf.argmax(mean_probs, axis=1)
  labels = tf.argmax(y_test, axis=1)
  accuracy = accuracy_score(labels.numpy(),
                            classification.numpy(),
                            normalize=True)
  print(accuracy)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_conv',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of samples per batch')
  parser.add_argument('--data', type=str, default='mnist_im',
                      help='data set to use')
  parser.add_argument('--model_type', type=str, default='lenet5',
                      help='type of model to use')
  parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs')
  parser.add_argument('--num_monte', type=int, default=10,
                      help='number of monte carlo samples for pred')

  args = parser.parse_args(sys.argv[1:])
  main(args)
