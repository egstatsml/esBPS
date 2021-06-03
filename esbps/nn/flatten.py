"""
Flattens data from conv array to dense array
"""
import tensorflow as tf

from tbnn.nn.layer import Layer


class FlattenLayer(Layer):
  """Flattens output of conv layers so can be used by MLP

  Attributes:
    config_data (Dict):
      config data supplied by JSON config sheet
    layer_num (int):
       which layer number this is in the network
  """
  def __init__(self, config_data, in_dims, layer_num):
    """set class attributes shared by all Conv layers"""
    self.layer_num = layer_num
    self.in_dims = in_dims
    self.out_dims = in_dims["width"] * in_dims["height"] * in_dims["channels"]
    self.layer_type = config_data["type"]
    self.name = config_data["name"]
    self.params = []
    

  def layer_forward_prop(self, input_):
    flattened = tf.contrib.layers.flatten(input_)
    return flattened
