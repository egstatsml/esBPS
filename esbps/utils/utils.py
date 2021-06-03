"""
Utility functions to help me load in some data,
save models etc.
"""
import cv2
import gzip
import os
import pickle
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt

class Data(object):
  """Will hold data set and some properties"""
  def __init__(self):
    self.x_train = []
    self.y_train = []
    self.x_val = []
    self.y_val = []
    self.x_test = []
    self.y_test = []
    self.num_train = 0
    self.num_test = 0
    self.num_val = 0
    self.x_test_orig = []
    self.label_dict = []
    self.dimension_dict = []
    self.batch_size = []

    

  def cast_data(self, cast_type):
    """will cast data to correct data type"""
    self.x_train = self.x_train.astype(cast_type)
    self.y_train =  self.y_train.astype(cast_type)
    self.x_val = self.x_val.astype(cast_type)
    self.y_val = self.y_val.astype(cast_type)
    self.x_test = self.x_test.astype(cast_type)
    self.y_test = self.y_test.astype(cast_type)
    if isinstance(self.x_test_orig, np.ndarray):
      self.x_test_orig = self.x_test_orig.astype(cast_type)

    

def normalise(input_):
  """
  normalise()

  Description:
  Will normalise data set to be within range of
  [-1, 1]
  """
  return (input_ - 0.5) * 2



def data_shape(data, label):
  """returns shape of data required for batch_training"""
  data_shape = [None]
  label_shape = [None]
  print(label.shape)
  print(label.shape[1])
  data_shape.extend(data.shape[1:])
  label_shape.append(label.shape[1])
  return (tf.TensorShape(data_shape), tf.TensorShape(label_shape))


def shuffle_data(input_, labels):
  """
  shuffle_data()

  Description:
  Will shuffle the input data and return
  a list of the data, where each element
  in the lists is a subset of the original
  unshuffled data.

  Args:
    INPUT_: np.array

  """
  labels = labels.reshape(-1, 1)
  indicies = np.random.permutation(labels.size).argsort()
  input_shuffle = np.take(input_, indicies, axis=0)
  labels_shuffle = np.take(labels, indicies, axis=0)
  return input_shuffle, labels_shuffle


def one_hot(labels, num_classes):
  """Creates one-hot encoding of labels"""
  labels_one = np.zeros((labels.size, num_classes))
  for ii in range(0, labels.size): # pylint: disable=invalid-name
    labels_one[ii, labels[ii]] = 1.0
  return labels_one.astype(np.float32)


def load_dataset(data_set, split):
  """Helper function to load in some standard data sets"""
  print(data_set.lower())
  if data_set.lower() == 'mnist':
    dataset = load_mnist(split)
  elif data_set.lower() == 'mnist_im':
    dataset = load_mnist_im(split)
  elif data_set.lower() == 'mnist_pad':
    dataset = load_mnist_pad(split)
  elif data_set.lower() == 'fashion_mnist':
    dataset = load_fashion_mnist(split)
  elif data_set.lower() == 'fashion_mnist_im':
    dataset = load_fashion_mnist_im(split)
  elif data_set.lower() == 'cifar_10':
    dataset = load_cifar_10(split)
  elif data_set.lower() == 'svhn':
    dataset = load_svhn(split)
  elif(data_set.lower() == 'toy_a'):
    dataset = load_toy_a()
  elif(data_set.lower() == 'toy_b'):
    dataset = load_toy_b()
  elif(data_set.lower() == 'toy_c'):
    dataset = load_toy_c()
  elif(data_set.lower() == 'test_a'):
    dataset = load_test_a()
  elif(data_set.lower() == 'test_b'):
    dataset = load_test_b()
  elif(data_set.lower() == 'linear'):
    dataset = load_linear()
  elif(data_set.lower() == 'linear_norm'):
    dataset = load_linear_norm()
  elif data_set.lower() == 'moons':
    dataset = load_moons()
  elif data_set.lower() == 'clusters':
    dataset = load_clusters()
  else:
    raise ValueError("Invalid data set supplied")
  dataset.num_train = dataset.y_train.shape[0]
  dataset.num_test = dataset.y_test.shape[0]
  dataset.num_val = dataset.y_val.shape[0]
  dataset.cast_data(np.float32)
  # add the number of training samples to the dimension dictionary as well
  dataset.dimension_dict['num_train'] = dataset.num_train
  return dataset


def load_toy_a():
  """Load toy data from Bayes by Backprop. paper

  Original function from
  "Weight Uncertainty in Deep Learning" by Blundell et al.
  https://arxiv.org/abs/1505.05424
  """
  dataset = Data()
  num_train = 500
  num_test = 1000
  epsilon = np.random.randn(num_train, 1) * 0.015
  dataset.x_train = np.sort(np.random.uniform(0, 0.5, num_train)).reshape(-1, 1)
  dataset.x_test = np.linspace(-0.2, 0.8, num_test).reshape(-1, 1)
  dataset.y_train = dataset.x_train +  0.3 * np.sin(2 * np.pi * (dataset.x_train + epsilon)) + (
    0.3 * np.sin(4 * np.pi * (dataset.x_train + epsilon)) + epsilon)
  dataset.y_test = dataset.x_test + 0.3 * np.sin(2 * np.pi * dataset.x_test) + (
    0.3 * np.sin(4 * np.pi * dataset.x_test))
  # now create the dictionary to hold the dimensions of our data
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                    'in_height' : 1, 'in_channels' : 1}
  # validation data is the same as the test data
  # no label_dict required for regression example
  return dataset



def load_toy_b():
  """Generate data with holes in it

  Original function taken from
  "Deep Exploration via Bootstrapped DQN" by Osband et al.
  https://arxiv.org/pdf/1602.04621.pdf
  In Appendix A
  """
  dataset = Data()
  num_train = 500
  num_test = 500
  epsilon = np.random.randn(num_train, 1) * 0.015
  dataset.x_train = np.concatenate((np.random.uniform(0.0, 0.6, np.int(num_train / 2)),
                            np.random.uniform(0.8, 1.0, np.int(num_train / 2))))
  # reshape to ensure is correct dims
  dataset.x_train = np.sort(dataset.x_train).reshape(-1, 1)
  dataset.x_test = np.linspace(-0.5, 1.5, num_test).reshape(-1, 1)
  dataset.y_train = (dataset.x_train +
                     np.sin(4.0 * (dataset.x_train + epsilon)) +
                     np.sin(13.0 * (dataset.x_train + epsilon)) + epsilon)
  dataset.y_test = (dataset.x_test + np.sin(4.0 * dataset.x_test) +
                    np.sin(13.0 * dataset.x_test))
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                    'in_height' : 1, 'in_channels' : 1}
  #validation data is the same as the test data
  return dataset



def load_toy_c():
  """non-linear function centered at x = 0

  Function from Yarin Gal's Blog
  http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html
  """
  dataset = Data()
  num_train = 500
  num_test = 500
  epsilon = np.random.randn(num_train, 1) * 0.015
  dataset.x_train = np.linspace(-0.45, 0.45, num_train).reshape(-1, 1)
  dataset.x_test = np.linspace(-0.6, 0.6, num_test).reshape(-1, 1)
  dataset.y_train = (dataset.x_train * np.sin(4.0 * np.pi * dataset.x_train) +
                     epsilon)
  dataset.y_test = dataset.x_test * np.sin(4.0 * np.pi * dataset.x_test)
  # now create the dictionary to hold the dimensions of our data
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                    'in_height' : 1, 'in_channels' : 1}
  #validation data is the same as the test data
  return dataset



def load_clusters(num_data=200, num_test=100):
  """creates clusters sampled from two Gaussians
  """
  dataset = Data()
  x_1 = np.random.multivariate_normal([0.0, 1.0], np.array([[1.0, 0.8], [0.8, 1.0]]), size=num_data//2)
  x_2 = np.random.multivariate_normal([2.0, 0.0], np.array([[1.0, -0.4], [-0.4, 1.0]]), size=num_data//2)
  X = np.vstack([x_1, x_2])
  y = np.ones(num_data).reshape(-1, 1)
  y[:np.int(num_data/2)] = 0
  print('X shape = {}'.format(X.shape))
  print('y shape = {}'.format(y.shape))
  scaler = MinMaxScaler((-1.0, 1.0)).fit(X)
  X = scaler.transform(X)
  # create a random split now
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=num_test)
  dataset.x_train = x_train
  dataset.x_test = x_test
  dataset.y_train = y_train
  dataset.y_test = y_test
  # now create the dictionary to hold the dimensions of our data
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  dataset.dimension_dict = {'in_dim' : 2, 'out_dim' : 1, 'in_width' : 1,
                            'in_height' : 1, 'in_channels' : 1}
  #validation data is the same as the test data
  return dataset



def load_moons(num_data=5000, num_test=100):
  """The make moons dataset from sklearn

  Function from Yarin Gal's Blog
  http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html
  """
  dataset = Data()
  num_train = 150
  num_test = 500
  X, Y = make_moons(noise=0.1, random_state=0, n_samples=num_data)
  Y = Y.reshape([-1, 1])
  scaler = MinMaxScaler((-1.0, 1.0)).fit(X)
  X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),
                                                      Y.astype(np.float32))
  dataset.x_train = scaler.transform(X_train)
  dataset.x_test = scaler.transform(X_test)
  dataset.y_train = y_train
  dataset.y_test = y_test
  # now create the dictionary to hold the dimensions of our data
  dataset.x_val = scaler.transform(X_test)
  dataset.y_val = y_test
  dataset.dimension_dict = {'in_dim' : 2, 'out_dim' : 1, 'in_width' : 1,
                    'in_height' : 1, 'in_channels' : 1}
  #validation data is the same as the test data
  return dataset



def load_mnist_old(split):
  """loads in mnist data in flattened array format"""
  file_p = gzip.open(os.path.join(
    os.environ['DATA_PATH'], 'mnist.pkl.gz'), 'rb')
  data_set_list = pickle.load(file_p, encoding='latin1')
  file_p.close()
  dataset = Data()
  #concatenate the data from the predefined train/val/test splits
  combined_x = np.concatenate((data_set_list[0][0],
                               data_set_list[1][0],
                               data_set_list[2][0]), axis=0)
  combined_y = np.concatenate((data_set_list[0][1],
                               data_set_list[1][1],
                               data_set_list[2][1]), axis=0)
  #centre data around zero in range of [-1, 1]
  combined_x = data_centre_zero(combined_x)
  #split the data sets how I want
  train_split, val_split, test_split = convert_split(split, combined_y.size)
  print('train_split = {}'.format(train_split))
  #seperate the data now
  print('combined type = {}'.format(type(combined_x)))
  print('combined = {}'.format(combined_x))

  dataset.x_train = combined_x[0:train_split, :]
  dataset.y_train = one_hot(combined_y[0:train_split], 10)
  dataset.x_val = combined_x[train_split: train_split + val_split, :]
  dataset.y_val = one_hot(combined_y[train_split: train_split + val_split], 10)
  test_start = train_split + val_split
  dataset.x_test = combined_x[test_start:test_start + test_split, :]
  dataset.y_test = one_hot(combined_y[test_start:test_start + test_split], 10)
  dataset.label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 784, 'out_dim' : 10,
                    'in_width' : 1, 'in_height' : 1, 'in_channels' : 1}
  print(np.max(dataset.x_train), np.min(dataset.x_train))
  return dataset




def load_mnist(split):
  """loads in mnist data in flattened array format"""
  dataset = Data()
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  print(x_train.shape)
  print(type(x_train))
  print(type(y_train))
  dataset.x_train = data_centre_zero(np.expand_dims(x_train, axis=-1).astype(np.float32))
  dataset.x_test = data_centre_zero(np.expand_dims(x_test, axis=-1).astype(np.float32))
  dataset.y_train = keras.utils.to_categorical(y_train, 10)
  dataset.y_test = keras.utils.to_categorical(np.array(y_test), 10)
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  dataset.dimension_dict = {'in_dim' : 784, 'out_dim' : 10, 'in_height': 28,
                    'in_width': 28, 'in_channels': 1}
  print(type(dataset.x_train))
  return dataset



def load_mnist_im(split):
  """load in mnist data in image format"""
  dataset = load_mnist(split)
  # now reshape all the X variables to 3-d arrays
  dataset.x_train = np.reshape(dataset.x_train,
                               (dataset.x_train.shape[0], 28, 28, 1))
  dataset.x_val = np.reshape(dataset.x_val,
                             (dataset.x_val.shape[0], 28, 28, 1))
  dataset.x_test = np.reshape(dataset.x_test,
                              (dataset.x_test.shape[0], 28, 28, 1))
  dataset.x_test_orig = np.copy(dataset.x_test)
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 784, 'out_dim' : 10, 'in_height': 28,
                    'in_width': 28, 'in_channels': 1}
  return dataset



def load_mnist_pad(split):
  """load in mnist data and pad with zeros to be 32x32

  This is what is done in the original LeNet paper
  """
  dataset = load_mnist_im(split)
  print('mnist before shape = {}'.format(dataset.x_train.shape))
  # now reshape all the X variables to 3-d arrays
  pad_sequence = ((0, 0), (2, 2), (2, 2), (0, 0))
  dataset.x_train = np.pad(dataset.x_train, pad_sequence, 'constant')
  dataset.x_val = np.pad(dataset.x_val, pad_sequence, 'constant')
  dataset.x_test = np.pad(dataset.x_test, pad_sequence, 'constant')
  dataset.x_test_orig = np.copy(dataset.x_test)
  print('mnist padded shape = {}'.format(dataset.x_train.shape))
  # now need to update the dictionary, as the dimensions are now different
  dataset.dimension_dict = {'in_dim' : 1024, 'out_dim' : 10, 'in_height': 32,
                    'in_width': 32, 'in_channels': 1}
  return dataset



def load_fashion_mnist(split):
  """loads in mnist data in flattened array format"""
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  dataset = Data()
  data_dir = os.path.join(os.environ['DATA_PATH'], 'fashion_mnist')
  # train_x_file_p = gzip.open(os.path.join(
  #   data_dir, 'train-images-idx3-ubyte.gz'), 'rb')
  # train_y_file_p = gzip.open(os.path.join(
  #   data_dir, 'train-labels-idx1-ubyte.gz'), 'rb')
  with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
    y_train = np.frombuffer(f.read(), dtype=np.uint8,
                           offset=8)
  with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz') ,'rb') as f:
    x_train = np.frombuffer(f.read(), dtype=np.uint8,
                            offset=16).reshape(len(y_train), 784)
  with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
    y_test = np.frombuffer(f.read(), dtype=np.uint8,
                           offset=8)
  with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz') ,'rb') as f:
    x_test = np.frombuffer(f.read(), dtype=np.uint8,
                            offset=16).reshape(len(y_test), 784)
  #centre data around zero in range of [-1, 1]
  dataset.x_train = data_centre_zero(x_train.astype(np.float32))
  dataset.y_train = one_hot(y_train, 10)
  dataset.x_test = data_centre_zero(x_test.astype(np.float32))
  dataset.y_test = one_hot(y_test, 10)
  # validation set not implemented for this dataset, so just
  # setting to same as testing data.
  # printed a big ole fat warning at the statr to remind me each time
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  dataset.label_dict = {0:'t-shirt', 1:'trouser', 2:'pullover',
                        3:'dress', 4:'coat', 5:'sandal', 6:'shirt',
                        7:'sneaker', 8:'bag', 9:'ankle boot'}
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 784, 'out_dim' : 10,
                    'in_width' : 1, 'in_height' : 1, 'in_channels' : 1}
  return dataset



def load_fashion_mnist_im(split):
  """load in mnist data in image format"""
  dataset = load_fashion_mnist(split)
  # now reshape all the X variables to 3-d arrays
  dataset.x_train = np.reshape(dataset.x_train,
                               (dataset.x_train.shape[0], 28, 28, 1))
  dataset.x_val = np.reshape(dataset.x_val,
                             (dataset.x_val.shape[0], 28, 28, 1))
  dataset.x_test = np.reshape(dataset.x_test,
                              (dataset.x_test.shape[0], 28, 28, 1))
  dataset.x_test_orig = np.copy(dataset.x_test)
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 784, 'out_dim' : 10, 'in_height': 28,
                    'in_width': 28, 'in_channels': 1}
  return dataset



def load_cifar_10(split):
  """load in cifar-10 data"""
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  dataset = Data()
  cifar_dir = os.path.join(os.environ['DATA_PATH'], 'cifar-10-batches-py/')
  data_files = [f for f in os.listdir(cifar_dir) if ('data_batch' in f)]
  train_data = {}
  for data_file in data_files:
    file_path = os.path.join(cifar_dir, data_file)
    with open(file_path, 'rb') as file_p:
      batch_data = pickle.load(file_p, encoding='latin1')
    # if this is the first batch we are loading in
    if('data' not in train_data):
      train_data = batch_data
      # otherwise append the new data to the other pre-loaded batches
      train_data['data'] = np.concatenate((train_data['data'],
                                           batch_data['data']), axis=0)
      train_data['labels'].extend(batch_data['labels'])

  # now load in the testing data
  print('cifar_dir = {}'.format(cifar_dir))
  test_path = os.path.join(cifar_dir, 'test_batch')
  with open(test_path, 'rb') as file_p:
    test_data = pickle.load(file_p, encoding='latin1')
  dataset.x_train, dataset.y_train, image_dg, min_, max_ = unpack_normalise_cifar(train_data)
  dataset.x_test, dataset.y_test, _, _, _ = unpack_normalise_cifar(test_data,
                                                                   image_dg,
                                                                   min_, max_)
  dataset.x_val, dataset.y_val, _, _,  _ = unpack_normalise_cifar(test_data,
                                                                  image_dg,
                                                                  min_, max_)
  dataset.x_test_orig = test_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  dataset.data_dict = {0:'airplane', 1:'automobile', 2:'bird',
               3:'cat', 4:'deer', 5:'dog',
               6:'frog', 7:'horse', 8:'ship',
               9:'truck'}
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 3072, 'out_dim' : 10, 'in_height': 32,
                    'in_width': 32, 'in_channels': 3}
  return dataset



def unpack_normalise_cifar(data, image_dg=None, min_=None, max_=None):
  # """unpack data from cifar format and normalise it"""
  images = data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  labels = data['labels']
  # # make sure the data is a float now
  print('max = {}'.format(np.max(images)))
  images = images.astype(np.float32)
  # now performing ZCA whitening
  # am using the ImageDataGenerator from Keras to do this, as it is built into
  # there. Should only be fit on the training data, and use the same
  # transformations on the test/val data.
  if image_dg == None:
    image_dg = keras.preprocessing.image.ImageDataGenerator(
      zca_whitening=True)
      #featurewise_center=True)
    #featurewise_std_normalization=False)
    image_dg.fit(images)
  # # now flow from there
  image_iter = image_dg.flow(images, batch_size=images.shape[0])
  pre_images = image_iter.next()
  # now apply global contrast normalization
  pre_images = global_contrast_normalization(images)
  # now subtract mean and divide by std.
  # if (min_ == None) or (max_ == None):
  #   min_ = np.min(pre_images)
  #   max_ = np.max(pre_images)
  #pre_images = data_centre_zero(pre_images, min_, max_)
  # pre_images, min_, max_ = data_normalise(pre_images, min_, max_)
  print('Stats after preprocessing')
  print('max = {}, min = {}, mean = {}, std = {}'.format(np.max(pre_images),
                                                         np.min(pre_images),
                                                         np.mean(pre_images),
                                                         np.std(pre_images)))
  return pre_images, one_hot(np.array(labels), 10), image_dg, min_, max_




def load_svhn(split):
  """load in preprocessed SVHN dataset

  refer to `load_svhn_and_preprocess` function for details on
  preprocessing
  """
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  dataset = Data()
  svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed')
  dataset.x_train = np.load(os.path.join(svhn_dir, 'x_train.npy'))
  dataset.x_test = np.load(os.path.join(svhn_dir, 'x_test.npy'))
  dataset.y_train = np.load(os.path.join(svhn_dir, 'y_train.npy'))
  dataset.y_test = np.load(os.path.join(svhn_dir, 'y_test.npy'))
  # load in the original data
  orig_svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn')
  dataset.x_test_orig = np.moveaxis(
    loadmat(os.path.join(orig_svhn_dir, 'test_32x32.mat'))['X'], 3, 0)
  plt.figure()
  plt.imshow(dataset.x_test_orig[0, ...].reshape(32, 32, 3))
  plt.savefig('test.png', cmap=None)
  # setting validation data to just be the same as the test data
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  # creating the label_dict
  dataset.label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 3072, 'out_dim' : 10, 'in_height': 32,
                    'in_width': 32, 'in_channels': 3}
  return dataset





def load_svhn_and_preprocess(split):
  """load in svhn data and performs preprocessing

  The SVHN dataset is in a Matlab format.
  Need to load it in using scipy.io.loadmat
  When loading it in, will be a dict with items 'X' for the input
  images and 'y' for the labels.
  """
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  dataset = Data()
  svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn')
  # load in the SVHN data into dict objects
  svhn_train = loadmat(os.path.join(svhn_dir, 'train_32x32.mat'))
  svhn_test = loadmat(os.path.join(svhn_dir, 'test_32x32.mat'))
  # extract the input data and preprocess
  dataset.x_train = svhn_preprocess(svhn_train['X'])
  dataset.x_test = svhn_preprocess(svhn_test['X'])
  # extract the label data and one-hot encode it
  # included minus one as svhn dataset is labelled from [1, 10]
  # want to convert to [0, 9]
  # currently zero is stored as class 10, so will just replace
  # all the entries with a 10 as a zero
  y_train = svhn_train['y']
  y_test = svhn_test['y']
  y_train[y_train == 10] = 0
  y_test[y_test == 10] = 0
  dataset.y_train = one_hot(y_train, 10)
  dataset.y_test = one_hot(y_test, 10)
  # setting validation data to just be the same as the test data
  dataset.x_val = dataset.x_test
  dataset.y_val = dataset.y_test
  # creating the label_dict
  dataset.label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
  # now create the dictionary to hold the dimensions of our data
  dataset.dimension_dict = {'in_dim' : 3072, 'out_dim' : 10, 'in_height': 32,
                    'in_width': 32, 'in_channels': 3}
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_train.npy'),
    dataset.x_train)
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_train.npy'),
    dataset.y_train)
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_test.npy'),
    dataset.x_test)
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_test.npy'),
    dataset.y_test)
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_val.npy'),
    dataset.x_val)
  np.save(
    os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_val.npy'),
    dataset.y_val)

  return dataset



def svhn_preprocess(svhn):
  """preprocessing of the street view house numbers dataset

  First reshape to correct dimensions and normalise.
  Then, further preprocessing is done using local contrast normalization
  in accordance with [1, 2].

  Args:
    svhn (np.array):
      street view house numbers dataset
  Returns:
    preprocessed svhn images

  References:
    [1] Sermanet, P., Chintala, S. and LeCun, Y., Convolutional Neural Networks
    Applied toHouse Numbers Digit Classification.
    [2] Zeiler, M. and Fergus, R., Stochastic Pooling for Regularization of Deep
    Convolutional Neural Networks
  """
  # swap the end axis to make the image index be the first dimension.
  # ie. go from [H, W, C, idx] to [idx, H, W, C]
  svhn = np.moveaxis(svhn, 3, 0)
  # now perform local contrast normalization
  svhn_normalized = local_contrast_normalization(svhn.astype(np.float32),
                                                       7, True)
  # apply global contrast normalization
  svhn_normalized = global_contrast_normalization(svhn_normalized)
  # now need to scale back to range [0, 1] so can convert back
  svhn_normalized = ((svhn_normalized - np.min(svhn_normalized)) /
                           (np.max(svhn_normalized) - np.min(svhn_normalized)))
  # just saving a test figure
  svhn = svhn
  plt.figure()
  plt.imshow(255 * svhn[0, ...].reshape([32, 32, 3]).astype(np.uint8))
  plt.savefig('./test.png')
  print('saved the figure')
  return svhn
  

def convert_color_space(data, conversion_code):
  """convert batch array images to new color space

  Args:
    data (np.array):
      4D array of images, with first dimension being batch
    conversion_code (opencv color code):
      Code to tell us how to perform the conversion (ie. cv2.COLOR_RGB2YUV)

  Returns:
    4D array with all images converted
  """
  # iterate over each image and convert to new color space
  print(data.shape)
  for i in range(0, data.shape[0]):
    data[i, ...] = cv2.cvtColor(data[i, ...].reshape(data.shape[1:]),
                                                     conversion_code)
  return data


def global_contrast_normalization(data, s=1, lmda=10, epsilon=1e-8):
  """ from
  https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python"""
  for img_idx in range(0, data.shape[0]):
    for channel_idx in range(0, data.shape[3]):
      X_average = np.mean(data[img_idx, :, :, channel_idx])
      X = data[img_idx, :, :, channel_idx] - X_average
      # `su` is here the mean, instead of the sum
      contrast = np.sqrt(lmda + np.mean(X**2))
      data[img_idx, :, :, channel_idx] = s * X / np.maximum(contrast, epsilon)
  return data



def local_contrast_normalization(data, radius, use_divisor, threshold=1e-6):
  """ Applies LeCun local contrast normalization [1]

  Original code taken from [2] and modified for tensorflow usage

  Args:
    data (np.array):
      4D image data in RGB format, and first dim being batch
    radius (int)
      determines size of Gaussian filter patch
    use_divisor (bool):
      whether or not to apply divisive normalization
    threshold (float):
      the threshold will be used to avoid division by zeros

  Returns:
    4D array with local contrast normalization applied to each image

  References:
    [1] Jarret, K. etal, What is the Best Multi-Stage Architecture for
        Object Recognition?
    [2] https://github.com/jostosh/theano_utils/blob/master/lcn.py
  """
  # Get Gaussian filter
  filters = gaussian_filter(radius, 3)
  # Compute the Guassian weighted average by means of convolution
  convout = tf.nn.conv2d(data, filters, 1, "SAME")
  # subract from the original data
  centered = data - convout
  # Boolean marks whether or not to perform divisive step
  if use_divisor:
    # Note that the local variances can be computed by using the centered_X
    # tensor. If we convolve this with the mean filter, that should give us
    # the variance at each point. We simply take the square root to get our
    # denominator
    # Compute variances
    sum_sqr = tf.nn.conv2d(tf.math.square(centered), filters, 1, "SAME")
    # Take square root to get local standard deviation
    denom = tf.math.sqrt(sum_sqr)
    # find the mean of each image for each channel
    per_img_mean = np.mean(denom, axis=(1,2))
    # reshape the mean to go across all pixels so can compare
    per_img_mean = np.repeat(per_img_mean[..., np.newaxis], 32, axis=2)
    per_img_mean = np.repeat(per_img_mean[..., np.newaxis], 32, axis=3)
    # now swap the axis so the channel axis is on the end again
    per_img_mean = np.moveaxis(per_img_mean, 1, 3)
    divisor = tf.math.maximum(
      per_img_mean, tf.reshape(denom, [-1, data.shape[1], data.shape[2], 3]))
    # Divisise step
    new_X = centered / tf.math.maximum(divisor, threshold)
  else:
    new_X = centered
  return np.array(new_X)


def gaussian_filter(radius, channels):
  """create a Gaussian filter to be applied to a batch of images

  Args:
    radius (int):
      square size of rhe kernel
    channels (int):
      the number of channels of Gaussian kernels

  Returns:
    Gaussian kernel of size [1, radius, radius, channels]
  """
  # creating a 2d Gaussian
  x, y = np.meshgrid(np.linspace(-1, 1, radius), np.linspace(-1 ,1 ,radius))
  d = np.sqrt(x * x + y * y)
  sigma, mu = 1.0, 0.0
  gaussian_2d = np.exp(- ((d-mu)**2 / ( 2.0 * sigma**2 )))
  print(gaussian_2d.shape)
  # repeat for all the channels
  gaussian_2d = gaussian_2d.reshape([radius, radius, 1])
  gaussian_3d = np.repeat(gaussian_2d[..., np.newaxis], channels, axis=3)
  print('here')
  print(gaussian_3d.shape)
  # then reshape to add a batch dimension of one, then return
  return gaussian_3d#reshape([1, *gaussian_3d.shape])




def load_test_a():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 10
  n_s = 10
  dataset = Data()
  dataset.x_train = np.linspace(-2, 2, n).reshape(-1,1)
  dataset.x_test = np.copy(dataset.x_train)
  dataset.y_train = dataset.x_train**2.0
  dataset.y_test = np.copy(dataset.y_train)
  dataset.x_val = dataset.x_train**2.0
  dataset.y_val = np.copy(dataset.y_train)
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                            'in_height' : 1, 'in_channels' : 1}
  return dataset




def load_test_b():
  """Simple function used during testing
  Mostly for testing event rate calculations, can compare
  directly with analytic solution for simple gaussian model
  """
  n = 1
  n_s = 1
  dataset = Data()
  dataset.x_train = np.array(1.0).reshape(1, 1)
  dataset.x_test = dataset.x_train
  dataset.y_train = np.array(1.5).reshape(1, 1)
  dataset.y_test = dataset.y_train
  dataset.x_val = dataset.x_train
  dataset.y_val = dataset.y_train
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                            'in_height' : 1, 'in_channels' : 1}
  return dataset



def load_linear():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 20
  n_s = 20
  dataset = Data()
  dataset.x_train = np.linspace(0, 5, n).reshape(-1,1)
  dataset.x_test = np.copy(dataset.x_train)
  dataset.y_train = 2.0 * dataset.x_train + 1.0 + np.random.randn(*dataset.x_train.shape)
  dataset.y_test = 2.0 * dataset.x_train + 1.0 + np.random.randn(*dataset.x_train.shape)
  dataset.x_val =  np.copy(dataset.x_train)
  dataset.y_val = 2.0 * dataset.x_train + 1.0 + np.random.randn(*dataset.x_train.shape)
  dataset.dimension_dict = {'in_dim' : 1, 'out_dim' : 1, 'in_width' : 1,
                            'in_height' : 1, 'in_channels' : 1}
  return dataset




def load_linear_norm():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 20
  n_s = 20
  dataset = load_linear()
  # will now scale it down

  return dataset





def batch_split(input_, output_, num_batches):
  """Split data into batches"""
  x_train = np.array_split(input_, num_batches)
  y_train = np.array_split(output_, num_batches)
  batch_size = y_train[0].size
  return x_train, y_train, batch_size



def save_model(sess, saver, epoch, model_name='model', model_dir='model'):
  """saves weights at current epoch"""
  # check the directory for where we are going to save exists
  # and if not, lets bloody make one
  check_or_mkdir(model_dir)
  saver.save(sess, os.path.join(model_dir, model_name),
             global_step=epoch)



def check_or_mkdir(arg):
  """Will make a directory if doesn't exist"""
  if not os.path.isdir(arg):
    os.mkdir(arg)



def var_dict_to_list(var_dict_list):
  """
  var_dict_to_list()

  Description:
  Will conver a list of dicts in the gradient


  var_list: list[dict{Tensors}] ::default=None
    list of dicts of trainable tensors in the graph
    One dict element in list for each layer
    eg. [{'w_sigma':w0_s, 'w_mu':w0_m},
    {'w_sigma':w1_s, 'w_mu':w1_m}, ... ]

  Returns:
    list[iterables]:

  """
  raise(NotImplementedError())


def convert_split(tvn_split, num_samples):
  """
  convert_split()

  Description:
  Will get the percentage values for split as specified
  by the input arg

  Args:
  tvn_split: (str)
    train_val_test_split
    command line arg that specifies the format for how we
    want to split the data (in percentages)
    ie. split = 80-10-10 for 80% test, 10% val and test
  num_samples: (int)
    the number of samples in this data set
  Returns:
    int(train_percent), int(val_percent), int(test_percent)
  """

  #seperate the input string by the slash locations
  str_split = tvn_split.split('_')
  #now check that 3 args were supplied
  if(len(str_split) != 3):
    raise(ValueError('Incorrect data split arg supplied: {}'.format(tvn_split)))
  #convert the split vals to strings
  train_split = int(str_split[0])
  val_split = int(str_split[1])
  test_split = int(str_split[2])
  #now check that they all sum up to 100
  #if not all of the data is used, print a warning
  if((train_split + val_split + test_split) < 100):
    print('WARNING, not all data used for experiment')
    print('train = {}%, val = {}%, test = {}%'.format(
      train_split, val_split, test_split))
  #if more than 100% data supplied, raise an exception
  elif((train_split + val_split + test_split) > 100):
    raise(ValueError(('Invalid split provided for data '
                      'does not sum to 100% \n'
                      'train = {}%, val = {}%, test = {}%'.format(
                        train_split, val_split, test_split))))
  #now change the percentage values to number of samples
  #from the data set
  train_split = np.int(num_samples * train_split/100)
  val_split = np.int(num_samples * val_split/100)
  test_split = np.int(num_samples * test_split/100)
  return train_split, val_split, test_split


def data_centre_zero(data, min_=None, max_=None):
  """centre data around zero"""
  if (min_==None) or (max_==None):
    min_ = np.min(data)
    max_ = np.max(data)
  # put in range of [0, 1]
  data = (data + np.abs(min_)) / (max_ + np.abs(min_))
  # now scale to range [-1, 1]
  return (data * 2.0) - 1.0#, min_, max_



def data_normalise(data, mean_=None, std_=None):
  """centre data around zero"""
  if (mean_ is None) or (std_ is None):
    mean_ = np.mean(data, axis=0)
    std_ = np.max(data, axis=0)
  # put in range of [0, 1]
  data = (data - mean_) / std_
  # now scale to range [-1, 1]
  return data, mean_, std_
