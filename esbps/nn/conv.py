"""
Implementation of vanilla conv net
"""
import abc
import six
import tensorflow as tf
from tensorflow import keras

from esbps.nn.mlp import MLP


@six.add_metaclass(abc.ABCMeta)
class Conv(MLP):
  """Base class for Convolutional Neural Network

  This class can be used to implement a generic conv.
  network, or it can be used as a parent class for BNNs using
  different distributions.

  Inherits from MLP which defines a few key attributes

  Attributes:
    num_conv (int):
      number of convolutional layers
    num_pool (int):
      number of pooling layers
    num_dense (int):
      number of dense layers
  """

  def __init__(self, prior, posterior,
               json_path=None, dimension_dict=None, name='Conv'):
    """Init to be Will be called from child class init method.

    Args:
      prior (string):
        type of prior to be placed over the weights
      posterior (string):
        type of posterior to be placed over the weights
      use_bias (bool):
        variable to say whether we want to use the bias variable
        in our networks or not
      json_path (string, default = None):
        path to JSON still config file
      dimension_dict (dict):
        info relating to dimension/size of data we are working with
    """
    self.num_conv = 0
    self.num_dense = 0
    self.num_pool = 0
    MLP.__init__(self, prior, posterior, json_path, dimension_dict)
    self.in_width = dimension_dict["in_width"]
    self.in_height = dimension_dict["in_height"]
    self.in_channels = dimension_dict["in_channels"]
    self.dims = [self.in_height, self.in_width, self.in_channels]

    

  def _initialise_network(self, data):
    """Private method that initialises variables in the graph.

    Args:
      data (dict):
        info from config file for network

    Returns:
      NA
    """
    # find the number of layers
    self.num_layers = len(data['layers'])
    self.num_conv = self._count_layers(data['layers'], 'conv')
    self.num_pool = self._count_layers(data['layers'], 'pool')
    self.num_dense = self._count_layers(data['layers'], 'dense')
    # now print this info
    print('Total of layers in network = {}'.format(self.num_layers))
    print('Number of conv layers = {}'.format(self.num_conv))
    print('Number of dense layers = {}'.format(self.num_dense))
    print('Layer information')
    print(data['layers'])
    for ii in range(0, self.num_layers):  # pylint: disable=invalid-name
      if(data["layers"][ii]["type"] == "conv"):
        self.layer.append(self.get_conv_layer(data["layers"][ii]))
      elif(data["layers"][ii]["type"] == "pool"):
        self.layer.append(
          self.get_pool_layer(data["layers"][ii]))
      elif(data["layers"][ii]["type"] == "flatten"):
        self.layer.append(
          self.get_flatten_layer())
      else:
        self.layer.append(
          self.get_dense_layer(data["layers"][ii]))



  def get_conv_layer(self, config_data):
    """get point estimate conv layer"""
    #setting posterior weight
    return keras.layers.Conv2D(
      config_data['conv_param']['num_output'],
      kernel_size=config_data['conv_param']['kernel_size'],
      strides=config_data['conv_param']['stride'],
      activation=config_data['activation'],
      kernel_initializer=config_data['kernel_initializer'],
      bias_initializer=config_data['bias_initializer'],
      padding='same')

  

  def get_dense_layer(self, config_data):
    """get point estimate dense layer"""
    return keras.layers.Dense(
      config_data['dense_param']['dim'],
      activation=config_data['activation'],
      use_bias=config_data['dense_param']['use_bias'],
      kernel_initializer=config_data['kernel_initializer'],
      bias_initializer=config_data['bias_initializer'])

  

  def get_flatten_layer(self):
    """get a flatten layer"""
    return keras.layers.Flatten()
  

  def get_pool_layer(self, config_data):
    """get a pooling layer"""
    return keras.layers.MaxPool2D(
      config_data['pooling_param']['kernel_size'],
      strides=config_data['pooling_param']['stride'])

  

  def _count_layers(self, layers, l_type):
    """ will count the number of layers of type l_type"""
    num_l = 0
    for layer in layers:
      if(layer["type"] == l_type):
        num_l = num_l + 1
    return num_l
