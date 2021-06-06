# esBPS

Implementation of efficient Stochastic Bouncy Particle Sampler (esBPS), a modified version of [1] to be suitable for application to Bayesian Neural Networks.


## Install

Will require you to have `tensorflow` installed. Refer to [intstall instructions here](https://www.tensorflow.org/install). This package is designed to utilise GPU acceleration and graph compilation, so is recommended to follow the GPU install instructions.

Once installed, run
```bash
pip install tensorflow_probability numpy matplotlib scikit-learn arviz
```

## Downloading Data

Data for toy problems is generated within, though data for CIFAR-10, MNIST and SVHN should be downloaded separately. This package is designed to have these data sets stored in a single directory, which can be found by setting an environment variable `DATA_PATH` to point to the parent directory containing the datasets.


### Creating Data Directory

Create the parent data directory wherever suits you best, and then create an environment variable to point there.

```bash
mkdir <path_to_data_dir>
export DATA_PATH=<path_to_data_dir>
```

### CIFAR-10
Download the CIFAR-10 python version from https://www.cs.toronto.edu/~kriz/cifar.html to your `DATA_PATH`, and then extract contents.
```bash
cd <path_to_data_dir>
tar -xvf cifar-10-python.tar.gz
```

### SVHN

Can download the `train_32x32.mat` and `test_32x32.mat` from [here](tar -xvf cifar-10-python.tar.gz). This should be downloaded to a directory named `svhn` in your `DATA_PATH`.

ie.
```bash
# make svhn directory
mkdir $DATA_PATH/svhn
# move the downloaded data to this directory
mv Downloads/train_32x32.mat $DATA_PATH/svhn
mv Downloads/test_32x32.mat $DATA_PATH/svhn
```

### Fashion-MNIST

Data can be downloaded [here](https://github.com/zalandoresearch/fashion-mnist#get-the-data). Download the train and test images and labels here, and similar to SVHN, move to a new independent directory.


```bash
# make Fashion-MNIST directory
mkdir $DATA_PATH/fashion_mnist
# move the downloaded data to this directory
mv Downloads/train-images-idx3-ubyte.gz $DATA_PATH/fashion_mnist
mv Downloads/t10k-labels-idx1-ubyte.gz $DATA_PATH/fashion_mnist
mv Downloads/t10k-images-idx3-ubyte.gz $DATA_PATH/fashion_mnist
mv Downloads/t10k-labels-idx1-ubyte.gz $DATA_PATH/fashion_mnist
```


## Running experiments

This package is designed to use `json` configuration files that contain model configuration info. Makes it easier to play around and experiment with different architectures. These config files are stored within `esbps/pdmp/json` directory.


### Regression examples
For all of these examples, you should set an output directory to where the samples from the MCMC will be saved, and all the plots.

There are three datasets described here that have been used in previous literature to demo performance of BNNs.

#### toy_a from [2]
```bash
mkdir `<toy_a_out>
/bin/test_bnn_regression.py ./esbps/pdmp/json/regression_small.json --out_dir <toy_a_out> --batch_size 100 --data toy_a --bps --num_burnin 1000 --num_results 2000 --no_log --ref 1.0
```

#### toy_b from [3]
```bash
mkdir `<toy_b_out>
/bin/test_bnn_regression.py ./esbps/pdmp/json/regression_small.json --out_dir <toy_b_out> --batch_size 100 --data toy_b --bps --num_burnin 1000 --num_results 2000 --no_log --ref 1.0
```

#### toy_c from [4]
```bash
mkdir `<toy_c_out>
/bin/test_bnn_regression.py ./esbps/pdmp/json/regression_small.json --out_dir <toy_c_out> --batch_size 100 --data toy_c --bps --num_burnin 1000 --num_results 2000 --no_log --ref 1.0
```


### Running convolutional network examples

For all of these examples, you should set an output directory to where the samples from the MCMC will be saved, and all the plots.

#### MNIST
```bash
mkdir `<mnist_out>
./bin/test_conv.py ./esbps/pdmp/json/lenet5.json --out_dir <mnist_out> --batch_size 1028 --data mnist_im --bps --num_burnin 1000 --num_results 2000 --no_log --ref 0.50
```

#### Fashion-MNIST
```bash
mkdir `<fashion_mnist_out>
./bin/test_conv.py ./esbps/pdmp/json/lenet5.json --out_dir <fashion_mnist_out> --batch_size 1028 --data mnist_im --bps --num_burnin 1000 --num_results 2000 --no_log --ref 0.50
```

#### SVHN
```bash
mkdir `<svhn_out>
./bin/test_conv.py ./esbps/pdmp/json/kaggle.json --out_dir <svhn_out> --batch_size 1028 --data mnist_im --bps --num_burnin 1000 --num_results 2000 --no_log --ref 0.50
```

#### CIFAR-10
```bash
mkdir `<cifar_out>
./bin/test_conv.py ./esbps/pdmp/json/kaggle.json --out_dir <cifar_out> --batch_size 1028 --data mnist_im --bps --num_burnin 1000 --num_results 2000 --no_log --ref 0.50
```


## References

[1] Pakman, A., Gilboa, D., Carlson, D., & Paninski, L. (2017, July). Stochastic bouncy particle sampler. In International Conference on Machine Learning (pp. 2741-2750). PMLR.

[2] Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015, June). Weight uncertainty in neural network. In International Conference on Machine Learning (pp. 1613-1622). PMLR.

[3] Osband, I., Blundell, C., Pritzel, A., & Roy, B. V. (2016, December). Deep exploration via bootstrapped DQN. In Proceedings of the 30th International Conference on Neural Information Processing Systems (pp. 4033-4041).

[4] Gal, Y. Uncertainty in Deep Learning, http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html
