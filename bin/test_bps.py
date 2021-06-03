import numpy as np
import tensorflow as tf
from tensorflow import test# import TestCase
import matplotlib.pyplot as plt

#from tensorflow_probability.python.internal import test_util# import TestCase
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from tbnn.pdmp.bps import BPSKernel, IsotropicGaussianBPSKernel


class TestBPS(test.TestCase):
  """Test the BPS kernel
  """

  def test_univariate_dist(self):
    """ Testing inference for univariate normal dist

    model is:
      p(theta) ~ Normal(0, 1)
      x = 1/N * sum(D)   # emperical mean of the data
      p(D|theta) ~ Normal(x|theta, Sigma)

    assume Sigma is known

    """
    # creating a very simple model normal model
    def MVN_data(mean, cov, n_samp=5):
      return np.random.multivariate_normal(mean.flatten(),
                                           cov,
                                           size=n_samp).astype(np.float32)

    def MVN_prior(size):
      return tfd.MultivariateNormalDiag(loc=np.zeros(size).astype(np.float32))

    def MVN_likelihood(q, cov):
      return tfd.MultivariateNormalFullCovariance(loc=q, covariance_matrix=cov)

    def potential(q, prior, data):
      likelihood = MVN_likelihood(q, cov)
      return 1.0 * (tf.reduce_mean(likelihood.log_prob(data)) + prior.log_prob(q))

    cov_array = np.array([[0.2, 0.0], [0.0, 0.2]]).astype(np.float32)
    cov = tf.constant(cov_array, dtype=tf.float32)
    prior = MVN_prior(2)
    data = MVN_data(np.array([0.5, -0.5]), cov_array, n_samp=50)
    print('data = {}'.format(data))
    # creating callable for the joint log prob
    joint_log_prob = lambda q: potential(q, prior, data)
    # now create the BPS kernel
    initial_state = np.array([1.0, 0.1]).astype(np.float32)
    print('joint_log_prob = {}'.format(joint_log_prob(initial_state)))
    kernel = BPSKernel(
      target_log_prob_fn=joint_log_prob,
      store_parameters_in_results=True,
      lambda_ref=0.10)
    # kernel = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
    #   target_log_prob_fn=joint_log_prob,
    #   num_leapfrog_steps=3,
    #   step_size=1.)
    # start sampling
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=50,
      current_state=initial_state,
      kernel=kernel)
    #      trace_fn=None,
    #      return_final_kernel_results=True,


  def test_compute_grad_l2_norm(self):
    """Test to make sure that the norm is calculated correctly"""
    # create a sample where the norm is known
    sample_grads = [np.ones(3) for x in range(0, 3)]
    known_l2_norm = 3.0
    # now creating an instance of the BPS kernel
    kernel = BPSKernel(
      target_log_prob_fn=lambda x: 1.0,
      store_parameters_in_results=True)
    l2_norm = kernel.compute_grad_l2_norm(sample_grads)
    # compare the known norm with that of the norm found using the fn
    self.assertAllClose(known_l2_norm, l2_norm, rtol=1e-8, atol=1e-8)


  def test_isotropic_gaussian(self):
    """ Testing inference for isotropic gaussian

    model is:
      x ~ Normal([1, 1], [[1, 0], [0, 1]])
    """
    def potential(q):
      print('q = {}'.format(q))
      return (tf.reduce_sum(tf.square(q)))
    # now create the BPS kernel
    initial_state = np.array([1.0, 0.0]).astype(np.float32)
    kernel = IsotropicGaussianBPSKernel(
      target_log_prob_fn=potential,
      store_parameters_in_results=True,
      lambda_ref=1.0)
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=initial_state,
      kernel=kernel)
    #      trace_fn=None,
    #      return_final_kernel_results=True,


    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.35)
    #plt.plot(samples[:, 0], samples[:, 1], c='k', alpha=0.15)
    plt.savefig('isotropic_gaussian_test.png')
    plt.savefig('isotropic_gaussian_test.pdf')
    plt.close()


  def test_sbps(self):
    """  Testing inference using linear rate approx. from SBPS paper

    Using Isotropic gaussian again
    model is:
      x ~ Normal([1, 1], [[1, 0], [0, 1]])
    """
    #tf.debugging.experimental.enable_dump_debug_info('./logs/debug',
    #                                                 tensor_debug_mode='SHAPE',
    #                                                 circular_buffer_size=-1)
    # from tensorflow.python.keras.backend import set_session
    # sess = tf.compat.v1.Session()
    # graph = tf.compat.v1.get_default_graph()

    # # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
    # # Otherwise, their weights will be unavailable in the threads after the session there has been set
    # set_session(sess)
    writer = tf.summary.create_file_writer('./logs')
    writer.set_as_default()
    @tf.function
    def graph_bps(*args, **kwargs):
      """Compile static graph for tfp.mcmc.sample_chain.
      Since this is bulk of the computation, using @tf.function here
      signifcantly improves performance (empirically about ~5x).
      """
      return tfp.mcmc.sample_chain(*args, **kwargs)

    def potential(q):
      #print('q = {}'.format(q))
      return (tf.reduce_sum(tf.square(q)))
    # now create the BPS kernel
    initial_state = np.array([1.0, 0.1]).astype(np.float32)
    kernel = BPSKernel(
      target_log_prob_fn=potential,
      store_parameters_in_results=True,
      lambda_ref=0.01)
    prev_kernel_results = kernel.bootstrap_results(initial_state)
    # # Call only one tf.function when tracing.
    #logdir = './logs/test_sbps'
    #writer = tf.summary.create_file_writer(logdir)
    #tf.summary.trace_on(graph=True, profiler=True)
    # a = graph_bps.get_concrete_function(num_results=1,
    #                                     current_state=initial_state,
    #                                     previous_kernel_results=prev_kernel_results,
    #                                     kernel=kernel, trace_fn=None)
    # print(type(a))
    # tf.io.write_graph(a, './logs', 'graph')
    #print(a)

    #print(sess.run(h))
    #writer.close()
    #try:
    samples = graph_bps(
      num_results=100,
      current_state=initial_state,
      previous_kernel_results=prev_kernel_results,
      kernel=kernel, trace_fn=None)
  # except Exception as e:
  #     print('something went wrong')
  #     print(e)
  #   print('writing')
    # with writer.as_default():
    #   tf.summary.trace_export(
    #     name="my_func_trace",
    #     step=0,
    #     profiler_outdir=logdir
    # )

    print(samples)

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2)
    plt.plot(samples[:, 0], samples[:, 1], c='k', alpha=0.15)
    plt.savefig('sbps_isotropic_gaussian_test.png')
    plt.close()




if __name__ == "__main__":
  test.main()
