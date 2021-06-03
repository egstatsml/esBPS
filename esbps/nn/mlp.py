"""
Base class for forming a Dense Neural Network
"""
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from esbps.nn import losses

# my user defined exceptions

class InvalidLossFuncError(ValueError):
  """Called if invalid loss string used"""
  def __init__(self, arg):
    print('Invalid loss function specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
      arg, ['bayes by backprop.', 'mc dropout', 'mle', 'map']))
    ValueError.__init__(self)


class InvalidLikelihoodError(ValueError):
  """Called if invalid likelihood string used"""
  def __init__(self, arg):
    print('Invalid likelihood specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
      arg, ['gaussian', 'softmax']))
    ValueError.__init__(self)


class InvalidOptimiserError(ValueError):
  """Called if invalid optimiser string used"""
  def __init__(self, arg):
    print('Invalid Optimiser specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
      arg, ['gradient descent', 'momentum', 'adam', 'rmsprop', 'adadelta']))
    ValueError.__init__(self)



class MLP(keras.Model):
  """Base class for Multi-Layer Peceptron Network

  Will define a list for prior and posterior over model
  parameters. Child classes will inherit from this
  class to set the type distributions to be used for the
  both the prior and posterior.

  Classes that inherit from this MUST IMPLEMENT the following abstract
  methods
    log_likelihood():
    log_gaussian_likelihood():
    get_layer():

  Attributes:
    prior_type (string): type of prior over weights
    posterior_type (string): type of posterior over weights
    loss_func (string): loss function to be used
    likelihood (string): type of likelihood over output
    num_layers (int): number of layers (hidden + output)
    in_dim (int): dimension of input data
    out_dim (int): dimension of output data
    num_classes (int): Number of classes
    layer (list(Layer)): list of MLP layer objects that inherit from the
      Layer abstract class
    use_bias (bool): whether we will use the bias vector on all layers
    learning_rate (float): learning rate for backprop
    optimiser (string): type of optimisation, as specified by config
    num_batches (int): number of batches to use, specified cmd. args.
    num_epochs (int): number of epochs, specified cmd. args.
  """

  def __init__(self, prior, posterior,
               json_path=None, dimension_dict=None, name='MLP'):
    """Init to be Will be called from child class init method.

    Args:
      prior (string):
        type of prior to be placed over the weights
      posterior (string):
        type of posterior to be placed over the weights
      json_path (string, default = None):
        path to JSON still config file
      dimension_dict (dict):
        info relating to dimension/size of data we are working with
    """
    super(MLP, self).__init__(self, name=name)
    self.prior_type = prior
    self.posterior_type = posterior
    self.prior_dict = None
    self.loss_func = []
    self.likelihood = []
    self.num_layers = 0
    self.in_dim = 0
    self.out_dim = 0
    self.in_width = 0
    self.in_height = 0
    self.in_channels = 0
    self.plot_dims = 0
    self.num_classes = 0  # will likely be the same as out_dim
    self.layer = []
    self.learning_rate = 0.0
    self.optimiser = []
    self.batch_size = 0
    self.num_batches = 0
    self.num_epochs = 0
    self.config_data = []
    if(json_path is not None) & (dimension_dict is not None):
      #initialise prior and posterior variables
      self.initialise(json_path, dimension_dict)



  def initialise(self, json_path, dimension_dict):
    """Parses JSON config. file

    Supplies parameters from config. file to those variables to private
    methods that will initialise the network

    Args:
      json_path (string):
        path to where JSON config file is
      dimension_dict (dict):
        info relating to dimension/size of data we are working with

    Returns:
      NA
    """
    with open(json_path) as file_p:
      data = json.load(file_p)
    # also save the config data
    self.config_data = data
    # try and set the prediction variable
    # if it isn't supplied in the config file, set to classify
    try:
      self.predict_type = data['predict']
    except(KeyError):
      self.predict_type = 'classify'
    # getting dimension of in/output from the config file
    self.in_dim = dimension_dict['in_dim']
    self.out_dim = dimension_dict['out_dim']
    self.in_width = dimension_dict["in_width"]
    self.in_height = dimension_dict["in_height"]
    self.in_channels = dimension_dict["in_channels"]
    self.plot_dims = self.format_plot_dims()
    self.num_classes = self.out_dim
    # reading other params
    self.likelihood = self._get_likelihood(data['likelihood'])
    self.loss_func = self._get_loss_func(data['loss'])
    self.learning_rate = np.float(data['learning rate'])
    self.optimiser = data['optimiser']
    self.num_epochs = np.uint32(data['epoch'])
    self.batch_size = np.uint32(data['batch'])
    self.num_batches = np.ceil(
      dimension_dict['num_train'] / self.batch_size).astype(float)
    print('batch size = {}'.format(self.batch_size))
    print('num train = {}'.format(dimension_dict['num_train']))
    print('num batches = {}'.format(self.num_batches))
    #see if there is a flag to tell us to use a pretrained network
    if('prior_file' in data):
      with open(data['prior_file'], 'rb') as file_p:
        self.prior_dict = pickle.load(file_p)

    self._initialise_network(data)


  

  def compute_output_shape(self, input_shape):
    print('setting output shape = {}'.format(self.out_dim))
    return self.out_dim



  def _initialise_network(self, data):
    """Private method that initialises variables in the graph.

    Args:
      data (dict):
        info from config file for network
    Returns:
      NA
    """
    #if a pretrained model was supplied,
    #load in the grap
    self.num_layers = len(data['layers'])
    print('Total of layers in network = {}'.format(self.num_layers))
    print('Layer information')
    print(data['layers'])
    for ii in range(0, self.num_layers):    # pylint: disable=invalid-name
      #setting parameters
      self.layer.append(self.get_layer(data["layers"][ii]))
          

  def _get_units(self, data, idx):
    """get the dimensions for the current layer

    Will check to see if a pre-trained layer was provided.
    If it was, will use those dimensions, otherwise will load them
    in from the JSON config file

    Args:
      data (dict):
        info from config file for network
      idx (int):
        layer index

    Returns:
      list with layer dimensions
    """
    if(data["layers"][idx]['prior_param']['prior_input'] != 'uninformative'):
      units = list(np.shape(
        self.prior_dict["{}/weight_loc".format(data["layers"][idx]['name'])]))
    else:
      units = np.int(data["layers"][idx]["dense_param"]["dim"])
    return units
  

  def call(self, inputs):
    """implements forward pass

    Point enstimate network

    Args:
    x: tf.placeholder
      placeholder for the input data

    returns:
    y: tf.Tensor
      containing output of network
    """
    # for the input data, say the output of some
    # pretend layer is our input data
    # just so we can use a single loop
    layer_out = inputs
    for layer in self.layers:
      # apply forward prop on the ii-1 layer to update the
      # output of the ii'th layer
      layer_out = layer(layer_out)
      print(layer_out.shape)
    return layer_out
    

  def layer_call(self, inputs, layer_idx):
    """forward pass up to certain layer

    Will return the output for the current model at
    layer `layer_idx`

    Args:
    inputs (np.array):
      input data
    layer_idx (int):
      index of the layer which will become our new output

    Returns:
      output at layer `layer_idx`
    """
    # for the input data, say the output of some
    # pretend layer is our input data
    # just so we can use a single loop
    layer_out = inputs
    # the +1 for the layer_idx is so that it is inclusive
    for layer in self.layer[:layer_idx+1]:
      # apply forward prop on the ii-1 layer to update the
      # output of the ii'th layer
      layer_out = layer(layer_out)
    return layer_out


  


  def build_cost(self):
    """Will build the cost function as specified by the config.

    The type of loss function to be used was saved during
    initialisation of the network

    Args:
      label (tf.Placeholder):
        Placeholder variable that will be filled with training
        labels
      output_ (tf.Operation):
        the output of the network

    Returns:
      Output of cost that we will use for optimisation
    """
    print('Loss Function used: {}'.format(self.loss_func))
    if(self.loss_func == 'mse'):
      cost = losses.mse_loss
    elif(self.loss_func == 'mle'):
      cost = losses.mle_loss
    elif(self.loss_func == 'map'):
      cost = losses.map_loss
    else:
      raise(InvalidLossFuncError(self.loss_func))
    return cost
    

  def log_likelihood(self, label, output_):
    """ return likelihood as specified in config. file"""
    if(self.likelihood == 'softmax'):
      log_likelihood = self.log_softmax_likelihood(label, output_)
    elif(self.likelihood == 'gaussian'):    # pylint: disable=no-else-return
      log_likelihood = self.log_gaussian_likelihood(label, output_)
    else:
      raise(NotImplementedError(self.likelihood))
    return log_likelihood
 

  def log_softmax_likelihood(self, label, output_):
    """using softmax likelihood at the moment"""
    softmax_likelihood = tf.reduce_sum(label * tf.nn.log_softmax(output_),
                                       axis=0)
    return tf.reduce_sum(softmax_likelihood)

  

  def log_gaussian_likelihood(self, label, output_, sigma=0.015):
    """Sample from log Gaussian likelihood

    Have removed the constant terms (such as the log(sigma)),
    specified by the assumption of the form of the likelihood
    as it makes the cost easier to track whilst training, and
    the gradient of these components is zero w.r.t model params

    Args:
      label (tf.Placeholder):
        tensor that will contain the labels
      output_ (tf.tensor):
        Output of the network (and will act as mean)
    sigma (float, default = 1.0):
        std for likelihood

    Returns:
      Sum of the log likelihood for this sample
    """
    #this commented out code is there to illustrate the complete log likelihood
    #log = (-0.5 * tf.math.log(2.0 * np.pi) - tf.math.log(sigma) -
    #       tf.square(label - output_) / (2.0 * tf.square(sigma)))
    #log likelihood that is used for training (removed constants)
    log = -1.0 * tf.divide(tf.square(label - output_), tf.square(sigma))
    return tf.reduce_sum(log)#tf.where(tf.is_nan(log), tf.zeros_like(log), log))

    

  def regularisation(self):
    """Regularisation needed to MAP Likelihood estimate

    Currenlty only works for Gaussian Likelihood, where
    L2 Regularisation is used for MAP
    """
    if(self.likelihood == "gaussian"):   # pylint: disable=no-else-return
      return self.l2_norm()
    # otherwise, raise an exception
    else:
      raise(NotImplementedError(
        "Only performing MAP estimate for Gaussian Likelihood"))

    

  def l2_norm(self):
    """Retern the L2 normalisation term"""
    return tf.reduce_mean(
      [tf.nn.l2_loss(layer.weight_loc) for layer in self.layer])

    

  def build_optimiser(self, global_step):
    """Will build the optimiser ops. as specified in the config file

    Will first compute the gradients, and will then clip them
    such that they aren't too great, which may be troublesome for
    large gradients in the scale (rho) parameters

    Args:
      cost (tf.operation):
        cost function to be minimised

    Returns:
      the operations to minimise the cost
    """
    if(self.optimiser == 'gradient descent'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    elif(self.optimiser == 'momentum'):
      optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.5)
    elif(self.optimiser == 'adam'):
      optimizer = keras.optimizers.Adam(self.learning_rate, clipnorm=1.0)
    elif(self.optimiser == 'rmsprop'):
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    elif(self.optimiser == 'adadelta'):
      optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
    else:
      raise InvalidOptimiserError(self.optimiser)

    return optimizer

  

  def _get_loss_func(self, arg):
    """Will check to see if supplied loss function is valid"""
    if((arg == 'bayes by backprop.') | (arg == 'mse') |
       (arg == 'mc dropout') | (arg == 'mle')): # pylint: disable=no-else-return
      return arg
    else:
      raise(InvalidLossFuncError(arg))
    

  def _get_likelihood(self, arg):
    """Will check to see if supplied likelihood is valid"""
    if((arg == 'softmax') | (arg == 'gaussian') |    # pylint: disable=no-else-return
       (arg == 'mle') | (arg == 'map')):
      return arg
    else:
      raise(InvalidLossFuncError(arg))
    

  def load_pretrained(self, pretrained):
    """Load a model that was pretrained to initialise our model

    Args:
    pretrained: string
      path to folder containing the pretrained weights
    """
    raise(NotImplementedError())
    

  def export_params(self, path):
    """will export trained parameters for the model

    Will get trained parameters from each layer and store them in a
    dictionary. The posterior type is then added to the dict, so that
    any future models reading in this data know how to deal with it.

    Args:
      path (str):
        where to save the parameters
    Returns:
      NA
    """
    params = {"type":self.posterior_type}
    for layer in self.layer:
      # get the dictionary with learnt parameters fromt this layers
      layer_params = layer.get_params()
      # add the layer params to the complete list
      for key, value in layer_params.items():
        params[key] = value

    # now save this to file using pickle
    with open(os.path.join(path, "params.pkl"), 'wb') as file_p:
      pickle.dump(params, file_p)

      

  def get_layer(self, config_data):
    """will get a Point estimate layer with current specs"""
    print('aaactivation = {}'.format(config_data['activation']))
    return keras.layers.Dense(
      config_data['dense_param']['dim'],
      activation=config_data['activation'],
      use_bias=config_data['dense_param']['use_bias'],
      kernel_initializer=config_data['kernel_initializer'],
      bias_initializer=config_data['bias_initializer'])



  def is_informative_prior(self, config_data):
    """checks to see if is an informative prior, returns boolean"""
    if('prior_param' in config_data):    # pylint: disable=no-else-return
      return config_data['prior_param']['prior_input'] != 'uninformative'
    else:
      return False

    

  def format_plot_dims(self):
    """ Formats the plotting dimensions based on our data set

    Want to know how to best plot our data at the end of the day
    """
    # if we are performing regression, just set the output to the
    # dimension of our output
    if(self.predict == 'regression'):
      return 1
    else:
      return [self.in_height, self.in_width, self.in_channels]
