"""
Functions to help us build and set up networks
"""
import json

from espbs.nn.mlp import MLP
from esbps.nn.conv import Conv

from esbps.pdmp.model import MCMCMLP


def get_tbnn(json_path, dimension_dict):
  """Factory for returning MLP class

  Will return a child object of the Network class,
  as determined by the JSON configuration file for the project.

  Args:
    json_path (string):
      path to JSON config file for project
    dimension_dict (dict):
      contains info relating to the dimension/size of our data
  Returns:
    Child object of Network class
  """
  with open(json_path) as file_p:
    data = json.load(file_p)
  if(data['posterior'] == 'point'):
    return MLP("point", "point", json_path,
               dimension_dict=dimension_dict)
  elif(data['posterior'] == 'conv point'):
    return Conv("point", "point", json_path,
                dimension_dict=dimension_dict)
  elif(data['posterior'] == 'dense mcmc'):
    return MCMCMLP("point", "point", json_path,
                   dimension_dict=dimension_dict)
  else:
    raise(NotImplementedError(
      """Currently only implemented point and MCMCMLP models"""))
