# flake8: noqa
"""
YellowFin optimizer for TensorFlow
Optimizer is based on:
https://github.com/JianGoForIt/YellowFin

We tried this experimental optimizer during the Planet Kaggle competition but
found that it performed worse than our other optimizer choices. It is unclear
why YF performed worse.
"""

from __future__ import print_function

import numpy as np
from math import ceil, floor
import tensorflow as tf
from tensorflow.python.training import momentum
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops

# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2

class YFOptimizer(object):
  def __init__(self, learning_rate=0.1, momentum=0.0, clip_thresh=None, beta=0.999, curv_win_width=20,
    mu_update_interval=1, zero_debias=True, delta_mu=0.0):
    '''
    clip thresh is the threshold value on ||lr * gradient||
    delta_mu can be place holder/variable/python scalar. They are used for additional
    momentum in situations such as asynchronous-parallel training. The default is 0.0
    for basic usage of the optimizer.
    Args:
      lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
      mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
      clip_thresh: python scalar. The cliping threshold for tf.clip_by_global_norm.
        if None, no clipping will be carried out.
      beta: python scalar. The smoothing parameter for estimations.
      delta_mu: for extensions. Not necessary in the basic use.
    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal learning rate
      in YellowFin. It is helpful when you want to do additional hand tuning
      or some decaying scheme to the tuned learning rate in YellowFin.
      Example on using lr_factor can be found here:
      https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
    '''
    self._lr = learning_rate
    self._mu = momentum

    self._lr_var = tf.Variable(learning_rate, dtype=tf.float32, name="YF_lr", trainable=False)
    self._mu_var = tf.Variable(momentum, dtype=tf.float32, name="YF_mu", trainable=False)
    # for step scheme or decaying scheme for the learning rates
    self.lr_factor = tf.Variable(1.0, dtype=tf.float32, name="YF_lr_factor", trainable=False)
    if clip_thresh is not None:
      self._clip_thresh_var = tf.Variable(clip_thresh, dtype=tf.float32, name="YF_clip_thresh", trainable=False)
    else:
      self._clip_thresh_var = None

    # the underlying momentum optimizer
    self._optimizer = \
      tf.train.MomentumOptimizer(self._lr_var * self.lr_factor, self._mu_var + delta_mu)

    # moving average for statistics
    self._beta = beta
    self._moving_averager = None

    # for global step counting
    self._global_step = tf.Variable(0, trainable=False)

    # for conditional tuning
    self._do_tune = tf.greater(self._global_step, tf.constant(0) )

    self._zero_debias = zero_debias

    self._tvars = None

    # for curvature range
    self._curv_win_width = curv_win_width
    self._curv_win = None


  def curvature_range(self):
    # set up the curvature window
    self._curv_win = \
      tf.Variable(np.zeros( [self._curv_win_width, ] ), dtype=tf.float32, name="curv_win", trainable=False)
    self._curv_win = tf.scatter_update(self._curv_win,
      self._global_step % self._curv_win_width, self._grad_norm_squared)
    # note here the iterations start from iteration 0
    valid_window = tf.slice(self._curv_win, tf.constant( [0, ] ),
      tf.expand_dims(tf.minimum(tf.constant(self._curv_win_width), self._global_step + 1), dim=0) )
    self._h_min_t = tf.reduce_min(valid_window)
    self._h_max_t = tf.reduce_max(valid_window)

    curv_range_ops = []
    with tf.control_dependencies([self._h_min_t, self._h_max_t] ):
      avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t] )
      with tf.control_dependencies([avg_op] ):
        self._h_min = tf.identity(self._moving_averager.average(self._h_min_t) )
        self._h_max = tf.identity(self._moving_averager.average(self._h_max_t) )
    curv_range_ops.append(avg_op)
    return curv_range_ops


  def grad_variance(self):
    grad_var_ops = []
    tensor_to_avg = []
    for t, g in zip(self._tvars, self._grads):
      if isinstance(g, ops.IndexedSlices):
        tensor_to_avg.append(tf.reshape(tf.unsorted_segment_sum(g.values, g.indices, g.dense_shape[0] ), shape=t.get_shape() ) )
      else:
        tensor_to_avg.append(g)
    avg_op = self._moving_averager.apply(tensor_to_avg)
    grad_var_ops.append(avg_op)
    with tf.control_dependencies([avg_op] ):
      self._grad_avg = [self._moving_averager.average(val) for val in tensor_to_avg]
      self._grad_avg_squared = [tf.square(val) for val in self._grad_avg]
    self._grad_var = self._grad_norm_squared_avg - tf.add_n( [tf.reduce_sum(val) for val in self._grad_avg_squared] )
    return grad_var_ops


  def dist_to_opt(self):
    dist_to_opt_ops = []
    # running average of the norm of gradeint
    self._grad_norm = tf.sqrt(self._grad_norm_squared)
    avg_op = self._moving_averager.apply([self._grad_norm,] )
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op] ):
      self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
      # single iteration distance estimation, note here self._grad_norm_avg is per variable
      self._dist_to_opt = self._grad_norm_avg / self._grad_norm_squared_avg
    # running average of distance
    avg_op = self._moving_averager.apply([self._dist_to_opt] )
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._dist_to_opt_avg = tf.identity(self._moving_averager.average(self._dist_to_opt) )
    return dist_to_opt_ops


  def after_apply(self):
    self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._beta, zero_debias=self._zero_debias)
    assert self._grads != None and len(self._grads) > 0
    after_apply_ops = []

    # get per var g**2 and norm**2
    self._grad_squared = []
    self._grad_norm_squared = []
    for v, g in zip(self._tvars, self._grads):
      if g is None: continue
      with ops.colocate_with(v):
        self._grad_squared.append(tf.square(g) )
    self._grad_norm_squared = [tf.reduce_sum(grad_squared) for grad_squared in self._grad_squared]

    # the following running average on squared norm of gradient is shared by grad_var and dist_to_opt
    avg_op = self._moving_averager.apply(self._grad_norm_squared)
    with tf.control_dependencies([avg_op] ):
      self._grad_norm_squared_avg = [self._moving_averager.average(val) for val in self._grad_norm_squared]
      self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
      self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)
    after_apply_ops.append(avg_op)

    with tf.control_dependencies([avg_op] ):
      curv_range_ops = self.curvature_range()
      after_apply_ops += curv_range_ops
      grad_var_ops = self.grad_variance()
      after_apply_ops += grad_var_ops
      dist_to_opt_ops = self.dist_to_opt()
      after_apply_ops += dist_to_opt_ops

    return tf.group(*after_apply_ops)


  def get_lr_tensor(self):
    lr = (1.0 - tf.sqrt(self._mu) )**2 / self._h_min
    return lr


  def get_mu_tensor(self):
    const_fact = self._dist_to_opt_avg**2 * self._h_min**2 / 2 / self._grad_var
    coef = tf.Variable([-1.0, 3.0, 0.0, 1.0], dtype=tf.float32, name="cubic_solver_coef")
    coef = tf.scatter_update(coef, tf.constant(2), -(3 + const_fact) )
    roots = tf.py_func(np.roots, [coef], Tout=tf.complex64, stateful=False)

    # filter out the correct root
    root_idx = tf.logical_and(tf.logical_and(tf.greater(tf.real(roots), tf.constant(0.0) ),
      tf.less(tf.real(roots), tf.constant(1.0) ) ), tf.less(tf.abs(tf.imag(roots) ), 1e-5) )
    # in case there are two duplicated roots satisfying the above condition
    root = tf.reshape(tf.gather(tf.gather(roots, tf.where(root_idx) ), tf.constant(0) ), shape=[] )
    tf.assert_equal(tf.size(root), tf.constant(1) )

    dr = self._h_max / self._h_min
    mu = tf.maximum(tf.real(root)**2, ( (tf.sqrt(dr) - 1)/(tf.sqrt(dr) + 1) )**2)
    return mu


  def update_hyper_param(self):
    assign_hyper_ops = []
    self._mu = tf.identity(tf.cond(self._do_tune, lambda: self.get_mu_tensor(),
      lambda: self._mu_var) )
    with tf.control_dependencies([self._mu] ):
      self._lr = tf.identity(tf.cond(self._do_tune, lambda: self.get_lr_tensor(),
        lambda: self._lr_var) )

    with tf.control_dependencies([self._mu, self._lr] ):
      self._mu = self._beta * self._mu_var + (1 - self._beta) * self._mu
      self._lr = self._beta * self._lr_var + (1 - self._beta) * self._lr
      assign_hyper_ops.append(tf.assign(self._mu_var, self._mu) )
      assign_hyper_ops.append(tf.assign(self._lr_var, self._lr) )
    assign_hyper_op = tf.group(*assign_hyper_ops)
    return assign_hyper_op


  def apply_gradients(self, grads_tvars, global_step=None):
    self._grads, self._tvars = zip(*[(g,t) for g,t in grads_tvars if g is not None])

    with tf.variable_scope("apply_updates"):
      if self._clip_thresh_var is not None:
        self._grads_clip, self._grads_norm = tf.clip_by_global_norm(self._grads, self._clip_thresh_var)
        apply_grad_op = \
          self._optimizer.apply_gradients(zip(self._grads_clip, self._tvars) )
      else:
        apply_grad_op = \
          self._optimizer.apply_gradients(zip(self._grads, self._tvars) )


    with tf.variable_scope("after_apply"):
      after_apply_op = self.after_apply()

    with tf.variable_scope("update_hyper"):
      with tf.control_dependencies( [after_apply_op] ):
        update_hyper_op = self.update_hyper_param()

    with tf.control_dependencies([update_hyper_op] ):
      self._increment_global_step_op = tf.assign(self._global_step, self._global_step + 1)

    return tf.group(apply_grad_op, after_apply_op, update_hyper_op, self._increment_global_step_op)


  def compute_gradients(self, loss, var_list, global_step=None,
                        gate_gradients=GATE_OP, aggregation_method=None,
                        colocate_gradients_with_ops=False, name=None,
                        grad_loss=None):
    return self._optimizer.compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
                                             aggregation_method=aggregation_method,
                                             colocate_gradients_with_ops=colocate_gradients_with_ops,
                                             grad_loss=grad_loss)


  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    """Adapted from Tensorflow Optimizer base class member function:
    Add operations to minimize `loss` by updating `var_list`.
    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `tf.gradients()` and `self.apply_gradients()` explicitly instead
    of using this function.
    """
    grads_and_vars = self._optimizer.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))
    for g, v in grads_and_vars:
      print("g ", g)
      print("v ", v)

    return self.apply_gradients(grads_and_vars)
