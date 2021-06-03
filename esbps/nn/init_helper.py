"""
Small functions that help parse in the data for
convolutional and dense layers. Saves from re-writing
a bunch of code for each of the different variants of layers,
and allows me to avoid things such as multiple inheritance which
can be a bit yucky.
"""

def format_init_config(config_data, dims):
  """helper function to parse/format layer params

  Will call either the dense or conv layer formatter as
  indicated by the layer type
  """
  if(config_data['type'] == 'dense'): # pylint: disable=no-else-return
    return dense_init_helper(config_data, dims)
  elif(config_data['type'] == 'conv'):
    return conv_init_helper(config_data, dims)
  else:
    raise(ValueError(('Invalid layer type {} should be either'
                      '\'dense\' or \'conv\'').format(config_data['type'])))

  

def dense_init_helper(config_data, dims):
  """Get dense layer params"""
  dense_config = {'in_dims' : dims[0],
                  'out_dims': dims[1],
                  'num_outputs': dims[0],
                  'use_bias' : config_data["dense_param"]["use_bias"],
                  'kernel_size' : 1, # not used in dense layer
                  'stride' : 1}      # not used in dense layer
  # the dims don't change for a dense network, use the precomuted dims
  # from MLP
  return dense_config, dims



def conv_init_helper(config_data, in_dims):
  """Get conv layer params

  For conv layers, the dimension parameter represents the dimension
  of the input data. The input + output dimensions will be used to
  determine the size of our kernel/weights
  """
  num_outputs = config_data["conv_param"]["num_output"]
  kernel_size = config_data["conv_param"]["kernel_size"]
  conv_config = {'in_dims' : in_dims,
                 'num_outputs' : num_outputs,
                 'out_dims': {"width":in_dims["width"],
                              "height":in_dims["height"],
                              "channels": num_outputs},
                 'use_bias' : config_data["conv_param"]["use_bias"],
                 'kernel_size' : kernel_size,
                 'stride' : config_data["conv_param"]["stride"]}
  # for conv layer, the in_dims argument tells us the dimension of the
  # input to this layer, and helps us form the actual dimensions for the
  # kernel/weights on this layer
  dims = [kernel_size, kernel_size, in_dims["channels"], num_outputs]
  return conv_config, dims
