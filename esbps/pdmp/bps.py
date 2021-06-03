"""
Implements BNN with BPS

Current implementation will parse the gradient functions,
the model and the data to the ARS module. This is for a first pass
to make sure everythiong works, and then tidy up.

To begin with the main bit of the implementation, start with the
grad_bounce_intensity_fn, and then work back all the way to the
log_likelihood etc.

Focus on methods from [1]


#### References
[1]:  Alexandre Bouchard-Côté∗, Sebastian J. Vollmer†and Arnaud Doucet;
      The Bouncy Particle Sampler: A Non-ReversibleRejection-Free Markov
      Chain Monte Carlo Method. https://arxiv.org/pdf/1510.02451.pdf

"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
import sys
import collections

from esbps.nn.mlp import MLP
from esbps.pdmp.poisson_process import IsotropicGaussianSampler, SBPSampler
from esbps.pdmp import utils



BPSKernelResults = collections.namedtuple(
  'BPSKernelResults',
  [
    'target_log_prob',        # For "next_state".
    'grads_target_log_prob',  # For "next_state".
    'velocity',               # also used for the next state
    'time',                   # also used for sampling from next state
    'acceptance_ratio'        # ratio for acceptence prob
  ])



class BPSKernel(tfp.mcmc.TransitionKernel):
  """Transition kernel for BPS within TFP

  Using description for UncalibratedHamiltonianMonteCarlo as the reference
  """
  def __init__(self,
               target_log_prob_fn,
               lambda_ref=1.0,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:

      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if seed is not None and tf.executing_eagerly():
      # TODO(b/68017812): Re-enable once TFE supports `tf.random_shuffle` seed.
      raise NotImplementedError('Specifying a `seed` when running eagerly is '
                                'not currently supported. To run in Eager '
                                'mode with a seed, use `tf.set_random_seed`.')
    if not store_parameters_in_results:
      # this warning is just here for the moment as a placeholder to remain
      # consistent with TFP, in case we need to store any results
      pass
    #self._seed_stream = SeedStream(seed, salt='bps_one_step')
    self._parameters = dict(
      target_log_prob_fn=target_log_prob_fn,
      grad_target_log_prob=grad_target_log_prob,
      lambda_ref=lambda_ref,
      bounce_intensity=bounce_intensity,
      grad_bounce_intensity=grad_bounce_intensity,
      state_gradients_are_stopped=state_gradients_are_stopped,
      seed=seed,
      name=name,
      store_parameters_in_results=store_parameters_in_results)
    self._momentum_dtype = None
    self.ipp_sampler = ipp_sampler(batch_size=batch_size, data_size=data_size)
    self.batch_size = batch_size
    self.data_size = data_size
    self.ar_sampler = []   # is initialised in the bootstrap_results method
    self.ref_dist = tfd.Exponential(self.lambda_ref)


  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  # need to define a setter attribute for the target_log_prob_fn
  @target_log_prob_fn.setter
  def target_log_prob_fn(self, target_log_prob_fn):
    self._parameters['target_log_prob_fn'] = target_log_prob_fn

  @property
  def grad_target_log_prob_fn(self):
    return self._parameters['grad_target_log_prob_fn']

  @property
  def bounce_intensity_fn(self):
    return self._parameters['bounce_intensity_fn']

  @property
  def grad_bounce_intensity_fn(self):
    return self._parameters['grad_bounce_intensity_fn']

  @property
  def state_gradients_are_stopped(self):
    return self._parameters['state_gradients_are_stopped']

  @property
  def lambda_ref(self):
    return self._parameters['lambda_ref']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @property
  def _store_parameters_in_results(self):
    return self._parameters['store_parameters_in_results']


  def one_step(self, previous_state, previous_kernel_results):
    """ performs update for Bouncy Particle Sampler

    main functionality described in Algorithm 1 of [1]
    For clarity, this implementation will adhere to the order of operations
    stated in Algorithm 1 in [1].

    Args:
      previous_state (tensor):
        previous state of parameters
      previous_kernel_results (`collections.namedtuple` containing `Tensor`s):
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `previous_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    #### References
    [1]:  Alexandre Bouchard-Côté∗, Sebastian J. Vollmer†and Arnaud Doucet;
          The Bouncy Particle Sampler: A Non-ReversibleRejection-Free Markov
          Chain Monte Carlo Method. https://arxiv.org/pdf/1510.02451.pdf
    """
    # main implementation of the BPS algorith from alg. 1. of reference [1]
    # step numbers are prepended with the same notation
    # starts at step 2., as assumes x and v have been initialised
    with tf.name_scope(mcmc_util.make_name(self.name, 'bps', 'one_step')):
      # preparing all args first
      # very similar to the HMC module in TFP
      # we are passing the values for target and the gradient from their
      # previous sample to ensure that the maybe_call_fn_and_grads won't
      # compute the target.
      # will be dooing it manually because we want to find the gradient w.r.t
      # parameters and with the time
      tf.print('start one step', output_stream=sys.stdout)
      [
        previous_state_parts,
        previous_velocity_parts,
        previous_target_log_prob,
        previous_target_log_prob_grad_parts,
      ] = self._prepare_args(
        self.target_log_prob_fn,
        previous_state,
        previous_kernel_results.velocity,
        target_log_prob=previous_kernel_results.target_log_prob,
        grads_target_log_prob=previous_kernel_results.grads_target_log_prob,
        maybe_expand=True,
        state_gradients_are_stopped=self.state_gradients_are_stopped)
      # (a) simulate the first arrival time bounce a IPP
      t_bounce, acceptance_ratio = self.ipp_sampler.simulate_bounce_time(
        self.target_log_prob_fn,
        previous_state_parts,
        previous_velocity_parts)
      # (b) Simulate ref time
      t_ref = tf.reshape(self.ref_dist.sample(1), ())
      tf.print('t_bounce = {}'.format(t_bounce))
      # (c) set the time and update to the next position
      time = tf.math.minimum(t_bounce, t_ref)
      next_state_parts = self.compute_next_step(previous_state_parts,
                                                previous_velocity_parts,
                                                time)
      # (d,e) sample the next velocity
      next_velocity_parts = self.compute_next_velocity(next_state_parts,
                                                       previous_velocity_parts,
                                                       time,
                                                       t_bounce,
                                                       t_ref)
      # now save the next state and velocity in the kernel results
      new_kernel_results = previous_kernel_results._replace(
        target_log_prob=previous_target_log_prob,
        grads_target_log_prob=previous_target_log_prob_grad_parts,
        velocity=next_velocity_parts,
        time=time,
        acceptance_ratio=acceptance_ratio)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(previous_state) else x[0]
      return maybe_flatten(next_state_parts), new_kernel_results


  def compute_next_step(self, state, velocity, time):
    """ updates the current state with the velocity and time found"""
    next_step = [u + v * time for u, v in zip(state, velocity)]
    return next_step


  def bounce_intensity(self, time, state_parts, velocity):
    time = tf.Variable(time, dtype=tf.float32)
    state_parts_velocity_time = [
      u + v * time for u,v in zip(state_parts, velocity)]
    target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      self.target_log_prob_fn, state_parts_velocity_time)
    # apply dot product between dU/dx and velocity
    bounce_intensity = utils.compute_dot_prod(grads_target_log_prob, velocity)
    return bounce_intensity.numpy()


  def grad_bounce_intensity(self, time, state_parts, velocity):
    """want to return lambda and d/dt{lambda}"""
    time = tf.Variable(time, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape_bounce:
      tape_bounce.watch([time])
      state_parts_velocity_time = [
        u + v * time for u,v in zip(state_parts, velocity)]
      target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, state_parts_velocity_time)
      # apply dot product between dU/dx and velocity
      bounce_intensity = utils.compute_dot_prod(grads_target_log_prob, velocity)
    # now compute gradient of above w.r.t time
    grad_t = tape_bounce.gradient(bounce_intensity, time)
    print(grads_target_log_prob)
    print('bounce_intensity = {}, time = {}, grad_t = {}'.format(bounce_intensity, time, grad_t))
    del tape_bounce
    return grad_t.numpy()


  def simulate_ref_time(self, time):
    return np.random.exponential() / self.lambda_ref
    #return np.random.exponential(self.lambda_ref)


  def compute_next_velocity(self,
                            next_state,
                            current_velocity_parts,
                            time,
                            t_bounce,
                            t_ref):
    """update the velocity based on the current times

    if t == t_bounce:
      update using the gradient of potential
      (Newtonian Elastic Collision)
    else if t == t_ref:
      refresh by sampling from a normal distribution

    Args:
      next_step (List(tf.Tensor)):
        the start of the next state in our updated trajectory
      current_velocity (List(tf.Tensor)):
        velocity just before the new updated trajectory
      time (tf.float):
        the time that will be used for the new trajectory
      t_bounce (tf.float):
        the proposed bounce time in our dynamics
      t_ref (tf.float):
        the sampled reference time

    Returns:
      the updated velocity for the next step
    """
    # if updating using a refresh
    print('t_ref = {}'.format(t_ref))
    refresh = lambda: self.refresh_velocity(current_velocity_parts)
    bounce = lambda: self.collision_velocity(next_state, current_velocity_parts)
    next_velocity = tf.cond(time == t_ref, refresh, bounce)
    return next_velocity


  def refresh_velocity(self, current_velocity_parts):
    """ Use refresh step for updating the velocity component

    This is just sampling from a random normal for each component
    Corresponds to step (d) in Alg 1 of [1]

    Args:
      current_velocity_parts (list(array)):
        parts of the current velocity

    Returns:
      Next velocity sampled from a Normal dist
    """
    tf.print('am refreshing')
    # normal = tfd.Normal(0., 1.0)
    # new_v = [normal.sample(x.shape) for x in current_velocity_parts]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]
    uniform = tfd.Uniform(low=-1.0, high=1.0)
    new_v = [uniform.sample(x.shape) for x in current_velocity_parts]
    new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    new_v = [v / new_norm for v in new_v]

    return new_v


  def collision_velocity(self, next_state, current_velocity_parts):
    """update the velocity based on simulated collision

    Collision is simulated using Newtonian Elastic Collision, as per
    equation (2) of ref [1].
    This equation is summarised here as.
    v_{i+1} = v - 2 * dot(grads_target, v) / norm(grads_target) * grads_target

    TODO: perhaps can use the already calculated for the target_log_prob
      here instead of recomputing in the _prepare_args fn when starting each
      iteration of the loop? Will save a forward pass, though not implementing
      now to keep thhings simple.

    Args:
      next_state (list(array)):
        next position for which we need to compute collision direction
      current_velocity_parts (list(array)):
        parts of the current velocity

    Returns:
      the updated velocity for the next step based on collision dynamics
    """
    tf.print('am bouncing')
    # need to compute the grad for the newest position
    _, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      self.target_log_prob_fn, next_state, name='bps_collision_update')
    # TODO: validate design choice here
    # Design choice: Since executing inner product, should we flatten everything
    # to a vector and do it that way, or just do element-wise multiplication and
    # then just sum? One way is mathematically more pretty, other might be
    # easier to write in code?
    # for the moment will do element-wise multiplication
    #
    # Need to find the Norm of the grad component, which requires looking at the
    # all the grad elements in the list.
    grads_norm = utils.compute_l2_norm(grads_target_log_prob)
    # now need to compute the inner product of the grads_target and the velocity
    dot_grad_velocity = utils.compute_dot_prod(grads_target_log_prob,
                                              current_velocity_parts)
    # can now compute the new velocity from the simulated collision
    new_v = [v - 2. * u * dot_grad_velocity / grads_norm for u, v in zip(
      grads_target_log_prob, current_velocity_parts)]
    return new_v


  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`.

    Returns an object with the same type as returned by one_step(...)

    Args:
      init_state (Tensor or list(Tensors)):
        initial state for the kernel
    Returns:
        an instance of BPSKernelResults with the initial values set
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'bps', 'bootstrap_results')):
      for x in init_state:
        print(x)
      init_state, _ = mcmc_util.prepare_state_parts(init_state)
      if self.state_gradients_are_stopped:
        init_state = [tf.stop_gradient(x) for x in init_state]
      [
          init_target_log_prob,
          init_grads_target_log_prob,
      ] = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn, init_state)
      # prepare the init velocity
      # do similar thing for the velocity components
      for x in init_state:
        print(x.shape)
      print(type(init_state))
      # get the initial velocity from the refresh fn, by just passing it the
      # state variables which will be used only to get the correct shape of things
      init_velocity = self.refresh_velocity(init_state)
      init_velocity, _ = mcmc_util.prepare_state_parts(init_velocity,
                                                       name='init_velocity')
      # initialise the AR sampler now
      if self._store_parameters_in_results:
        return BPSKernelResults(
          target_log_prob=init_target_log_prob,
          grads_target_log_prob=init_grads_target_log_prob,
          velocity=init_velocity,
          time=1.0,
          acceptance_ratio=0.0)
      else:
        raise NotImplementedError('need to implement for not saving results')

  @tf.function
  def examine_event_intensity(self, state, velocity, time):
    """method used to examine the event intensity

    This is intended as a diagnositc function; it isn't integral to
    the actual implementation of the BPS, but it is for how the
    intensity function for the IPP that controls the event rate
    behaves.

    Is intended to be run in a loop to see how the intensity changes
    w.r.t. time.

    Args:
      state (Tensor or list(Tensors)):
        state for the kernel
      velocity (Tensor or list(Tensors)):
        current velocity for the kernel
      time (tf.float32):
        the time we are currently looking at

    Returns:
      eval of the IPP rate for the BPS for current state, velocity and time.
    """
    # update the state for givin time and velocity
    updated_state = [s + v * time for s,v in zip(state, velocity)]
    # need to get the gradient of the current state
    _, grad = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn,
                                                updated_state)
    print('grad = {}'.format(grad))
    ipp_intensity = utils.compute_dot_prod(grad, velocity)
    return ipp_intensity


  def _prepare_args(self, target_log_prob_fn,
                    state,
                    velocity,
                    target_log_prob=None,
                    grads_target_log_prob=None,
                    maybe_expand=False,
                    state_gradients_are_stopped=False):
    """Helper which processes input args to meet list-like assumptions.

    Much of this is directly copied from the HMC module in TFP. Have updated
    for BPS
    """
    state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
    # do similar thing for the velocity components
    velocity_parts, _ = mcmc_util.prepare_state_parts(velocity,
                                                      name='current_velocity')
    #acceptance_ratio = 0.0
    if state_gradients_are_stopped:
      state_parts = [tf.stop_gradient(x) for x in state_parts]
    print('prepare args before grad target log prob = {}'.format(grads_target_log_prob))
    target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
    print('prepare args grads target log prob = {}'.format(grads_target_log_prob))
    print('prepare args target log prob = {}'.format(target_log_prob))
    def maybe_flatten(x):
      return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
    # returning the original state, as it is needed for computing the next
    # state/position in step (c) of Alg. 1 from [1]
    return [
      maybe_flatten(state_parts),
      maybe_flatten(velocity_parts),
      target_log_prob,
      grads_target_log_prob,
    ]

      

class IsotropicGaussianBPSKernel(BPSKernel):
  """Transition kernel for BPS for Isotropic Gaussian Target

  Mostly the same as the BPS, but uses analytic results for sampling using
  the inversion method.
  """
  @mcmc_util.set_doc(BPSKernel.__init__.__doc__)
  def __init__(self,
               target_log_prob_fn,
               lambda_ref=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    super().__init__(target_log_prob_fn,
                     lambda_ref=lambda_ref,
                     grad_target_log_prob=grad_target_log_prob,
                     bounce_intensity=bounce_intensity,
                     grad_bounce_intensity=grad_bounce_intensity,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     seed=seed,
                     store_parameters_in_results=store_parameters_in_results,
                     name=name)
    self.ipp_sampler = IsotropicGaussianSampler()


class IterBPSKernel(BPSKernel):
  """Transition kernel for BPS that handles an iter object
  to parse mini batches of data

  Is different to the normal kernels in BPS, in that their is no
  target_log_prob_fn. There is parent function that is called upon each
  iteration, which first iterates over the next batch of data within
  the model, and then returns to local target_log_prob_fn.

  An example would be,

  ```python
  def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
    def _fn():
      X, y = dataset_iter.next()
      return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
    return _fn

  # create the kernel now
  kernel = IterBPSKernel(iter_bnn_neg_joint_log_prob, ...)
  # include any other args that are parsed to the normal BPSKernel
  ```
  This is different to the BPSKernel (or any other kernel in TFP)
  in that they would just have something like,

  ```python
  # create a callable of the neg joint log prob
  target_log_prob = bnn_neg_joint_log_prob_fn(model, weight_prior_fns,
                                              bias_prior_fns, X, y)
  # create the kernel now
  kernel = BPSKernel(target_log_prob, ...)
  # similarly handle any other args for normal BPS
  ```
  """
  def __init__(self,
               parent_target_log_prob_fn,
               lambda_ref=1.0,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      parent_target_log_prob_fn: Python callable returns another Python callable
        which then takes an argument like `current_state`
        (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Refer to the class docstring for info and examples about how this
        differs from the normal kernel construction.
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    # need to initialise the target_log_prob_fn, so that the
    # bootstrap_results method can be called.
    self.parent_target_log_prob_fn = parent_target_log_prob_fn
    target_log_prob_fn = self.parent_target_log_prob_fn()
    super().__init__(target_log_prob_fn,
                     lambda_ref,
                     ipp_sampler,
                     batch_size,
                     data_size,
                     grad_target_log_prob,
                     bounce_intensity,
                     grad_bounce_intensity,
                     state_gradients_are_stopped,
                     seed,
                     store_parameters_in_results,
                     name)


  def one_step(self, previous_state, previous_kernel_results):
    """Will call the parent_target_log_prob_fn, which will get the
    next instance of the target_log_prob fn and set it. Will than call
    the parent class to perform a one step on this iteration.
    """
    # get the new local target_log_prob_fn
    self._parameters['target_log_prob_fn'] = self.parent_target_log_prob_fn()
    print('target_log_prob_fn = {}'.format(self.target_log_prob_fn))
    next_state_parts, next_kernel_results = super().one_step(previous_state, previous_kernel_results)
    return next_state_parts, next_kernel_results
