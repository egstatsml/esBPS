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
from tbnn.pdmp.bps import IterBPSKernel

import argparse
import sys
import os
from glob import glob
import pickle


def main(args):
  # can't store everything in memory if the model is large enough,
  # so will select which vars we want to hang onto here
  # the layer iter will specify which params we should look at from which
  # "layer", but will be such that is:
  # 0 = kernel_layer_1, 1 = bias_layer_1, 2 = kernel_layer_2 etc.
  layer_iter = 2
  param_idx = [2,0,0,0]
  # iterate over the chain files in the path and load them in
  # get a list of all the chain files
  chain_files = glob(os.path.join(args.in_dir, 'chain*.pkl'))
  # create a list to store all samples from predictive posterior
  chain_list = []
  printed_out = False
  for chain_file in chain_files:
    with open(chain_file, 'rb') as f:
      chain = pickle.load(f)
    # just print the shape of everything so that we know what we are looking at
    if(not printed_out):
      printed_out = False
      for i in range(0, len(chain)):
        print('shape iter {} = {}'.format(i, chain[i].shape))
    chain_list.append(chain)

  combined_chain = []
  combined_single_chain = []
  for i in range(0, len(chain_list[0])):
    combined_chain.append(np.concatenate([x[i] for x in chain_list], axis=0))
  print(combined_chain)
  #combined_chain = [np.concatenate(chain_list, axis=0)
  print(type(combined_chain))
  for x in combined_chain:
    print(np.shape(x))
    print(type(x))
  # want to get correlation matrix for input layer
  # input layer of dimension [num_samples, 1, n_hidden]
  # will get rid of the [1] dimension so can calculate
  # correlation
  input_params = np.squeeze(combined_chain[0])
  corr = np.corrcoef(input_params, rowvar=False)
  print(corr.shape)
  plt.figure()
  plt.matshow(corr)
  plt.savefig(os.path.join(args.out_dir, 'corr.png'))
  plt.savefig(os.path.join(args.out_dir, 'corr.pdf'))

  # now lets create a plot of density estimate
  plt.figure()
  az.plot_density(input_params[:, 0], figsize=(6, 2))
  plt.title('')
  plt.savefig(os.path.join(args.out_dir, 'dist.png'), bbox_inches='tight')
  plt.savefig(os.path.join(args.out_dir, 'dist.pdf'))



  #chain = np.array(combined_chain)
  #print(chain.shape)

  #chain_az = az.from_tfp(posterior=combined_chain)
  #print(chain_az.posterior)
  #single_var = combined_chain[2][:,2,0,0,0]
  #az.plot_trace(single_var)
  #plt.scatter(45 * np.arange(1, 22), [-0.16] * 21)
  #plt.savefig(os.path.join(args.out_dir, 'trace.png'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_conv',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('in_dir', type=str,
                      help='directory with the chain files')
  parser.add_argument('--out_dir', type=str, default='./out',
                      help='out directory where data is saved')
  args = parser.parse_args(sys.argv[1:])
  main(args)
