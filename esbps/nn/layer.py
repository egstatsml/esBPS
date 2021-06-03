"""
Abstract base class for a layer in a neural network.
Any type of layer shoulf inherit from this class
"""
import abc
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras

class InvalidActivationError(ValueError):
  """Exception for incorrect activation specification"""
  def __init__(self, arg):
    print('Invalid activation function specified in config file')
    print('Supplied {}, available activation functions are {}'.format(
      arg, ['ReLU', 'Leaky_ReLU', 'Tanh', 'Sigmoid', 'Linear']))
    ValueError.__init__(self)



class Layer(keras.layers.Layer):
  """Abstract class for layer in network

  Any class that is used to define a layer in an MLP should inherit from
  this class. This class doesn't do much, but it is here mainly for
  consistency in code design.

  The following abstract methods must be overwritten in child classes:
    layer_forward_prop()


  Attributes:
    use_bias (bool):
      say whether we are using bias vector for this layer
    layer_num (int):
       which layer number this is in the network
    dims (list[int]):
      Dimensions of this layer
  """
  def __init__(self, config_data, dims, layer_num, params):
    """Initialise class variables

    Will in initialise variables used be Dense and Conv layers,
    and then will initialise
    """
    self.use_bias = params['use_bias']
    self.in_dims = params['in_dims']
    self.out_dims = params['out_dims']
    self.use_bias = params['use_bias']
    self.num_outputs = params['num_outputs']
    self.dims = dims
    self.layer_num = layer_num
    self.activation = config_data["activation"]
    self.layer_type = config_data["type"]
    self.name = config_data["name"]
    self.params = []
    # following two parameters not used in dense layers
    # they will be set to one in dense layers
    self.kernel_size = params['kernel_size'] # only used in conv layers
    self.stride = params['stride']           # only used in conv layers
    
    

  def apply_activation(self, tens):
    """ Apply activation on current layer"""
    if(self.activation == "ReLU"):   # pylint: disable=no-else-return
      return tf.nn.relu(tens)
    elif(self.activation == "Leaky_ReLU"):
      return tf.nn.leaky_relu(tens)
    elif(self.activation == "Tanh"):
      return tf.nn.tanh(tens)
    elif(self.activation == "Sigmoid"):
      return tf.nn.sigmoid(tens)
    elif(self.activation == "Linear"):
      return tens
    else:
      raise InvalidActivationError(self.activation)

    

  def get_layer_var_names(self):
    """returns parameter names used in tensorboard"""
    return(self.params)

  

  def get_params(self):
    """will export trained parameters

    If a child class/layer has learnt parameters, it should
    overwrite this method. If it doesn't, will use this
    method and will just return an empty dictionary
    """
    return {}

  

  @abc.abstractmethod
  def layer_forward_prop(self):
    """abstract method for forward prop"""
    raise NotImplementedError("Must be created in child classes")

   

  def _orthogonal_init(self):
    """Orthogonal kernel initialisation

    Following results of Saxe et al (2014)
    https://arxiv.org/pdf/1312.6120.pdf
    Initialises kernel with weights such that when they are
    transformed to a matrix to perform forward/back prop.,
    the rows become orthogonal.

    Will work for both conv and dense connected layers.

    Following implementation similar to that in Lasagne
    https://github.com/Lasagne/Lasagne/commit/
      477eec1f5babc552c849115b6276eb81f585dac5

    Args:
      NA

    Returns:
      np.array with the elements that result in orthogonal rows
    """
    # if is a conv layer, will need to reshape to fan in matrix,
    # which is of dimension
    # num input feature maps * filter height * filter width
    if(len(self.dims) > 2):
      rv_samp = np.random.randn(self.dims[2],
                          self.dims[0] * self.dims[1] * self.dims[3])
      out_sigma = np.sqrt(1.0 / rv_samp.shape[1])
      # otherwise will be a densely connected layer
    else:
      rv_samp = np.random.randn(self.dims[0], self.dims[1])
      out_sigma = np.sqrt(1.0 / rv_samp.shape[0])
    # perform SVD
    U, _, V = np.linalg.svd(rv_samp, full_matrices=False) #pylint: disable=invalid-name
    # both U and V are orthoginal matricies, so will choose the one
    # that is the correct dimensions for our layer
    ortho_matrix = U if U.shape == rv_samp.shape else V
    # rescale so it is unit variance for each vector
    # print("std(q) = {}".format(np.std(q)))
    ortho_norm = (ortho_matrix / np.std(ortho_matrix)) * out_sigma
    #print("std(qs) = {}".format(np.std(qs)))
    #print(q.shape)
    return ortho_norm.reshape(self.dims).astype(np.float32)

