#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import h5py

class linear_system(object):
  def __init__(self, physical_system, f_background, linearized_source_sink_term):
    self.physical_system = physical_system

    # Storing domain information from physical system:
    # Getting resolution and size of configuration and velocity space:
    self.N_q1, self.q1_start, self.q1_end = physical_system.N_q1, physical_system.q1_end, physical_system.q1_end
    self.N_q2, self.q2_start, self.q2_end = physical_system.N_q2, physical_system.q2_end, physical_system.q2_end
    self.N_p1, self.p1_start, self.p1_end = physical_system.N_p1, physical_system.p1_end, physical_system.p1_end
    self.N_p2, self.p2_start, self.p2_end = physical_system.N_p2, physical_system.p2_end, physical_system.p2_end
    self.N_p3, self.p3_start, self.p3_end = physical_system.N_p3, physical_system.p3_end, physical_system.p3_end

    # Evaluating step size:
    self.dq1 = physical_system.dq1 
    self.dq2 = physical_system.dq2
    self.dp1 = physical_system.dp1
    self.dp2 = physical_system.dp2
    self.dp3 = physical_system.dp3

    # Getting number of ghost zones, and the boundary conditions that are utilized
    self.N_ghost               = physical_system.N_ghost
    self.bc_in_x, self.bc_in_y = physical_system.in_x, physical_system.in_y 

    self.p1_center, self.p2_center, self.p3_center = self._calculate_p()

    self._source_or_sink = self.linearized_source_sink_term
  
  def compute_moments(self, moment_name):
    try:
      moment_exponents = np.array(self.physical_system.moments[moment_name])
    except:
      raise KeyError('moment_name not defined under physical system')

    try:
      moment_variable = 1
      for i in range(moment_exponents.shape[0]):
        moment_variable *= self.p1_center**(moment_exponents[i, 0]) + \
                           self.p2_center**(moment_exponents[i, 1]) + \
                           self.p3_center**(moment_exponents[i, 2])
    except:
      moment_variable  = self.p1_center**(moment_exponents[0]) + \
                         self.p2_center**(moment_exponents[1]) + \
                         self.p3_center**(moment_exponents[2])

    moment = np.sum(np.sum(np.sum(self.f * moment_variable, 2)*self.dp3, 1)*self.dp2, 0)*self.dp1
    return(moment)

  def dump_variables(self, file_name, **args):
    h5f = h5py.File(file_name + '.h5', 'w')
    for variable_name in args:
      h5f.create_dataset(str(variable_name), data = variable_name)
    h5f.close()
    
    return

  def dump_distribution_function_5D(self, file_name, params):
    """
    Used to create the 5D distribution function array from the 3V delta_f_hat
    array. This will be used in comparison with the solution as given by the
    nonlinear method.
    """  
    q1_center = self.q1_start + (0.5 + np.arange(0, self.N_q1, 1)) * self.dq1
    q2_center = self.q2_start + (0.5 + np.arange(0, self.N_q2, 1)) * self.dq2

    q2_center, q1_center = np.meshgrid(q2_center, q1_center)
    
    q1_center, q2_center = q1_center.reshape([self.N_q1, self.N_q1, 1, 1, 1]),\
                           q2_center.reshape([self.N_q1, self.N_q1, 1, 1, 1])
    
    f_dist = np.zeros([self.N_q1, self.N_q2, self.N_p1, self.N_p2, self.N_p3])
    
    # Converting delta_f_hat --> delta_f
    f_dist = (self.delta_f_hat.reshape([1, 1, self.N_p1, self.N_p2, self.N_p3]) * \
              np.exp(1j*params.k_q1*q1_center + 1j*params.k_q2*q2_center)).real

    # Adding back the background distribution(delta_f --> delta_f + f_background):
    f_dist += np.tile(self.f_background.reshape(1, 1, self.N_p1, self.N_p2, self.N_p3),\
                      (self.N_q1, self.N_q2, 1, 1, 1)
                     ) 
    
    h5f = h5py.File(file_name + '.h5', 'w')
    h5f.create_dataset('distribution_function', data = f_dist)
    h5f.close()
    
    return

  def dY_dt(self, Y0):
    """
    Returns the value of the derivative of the mode perturbation of the distribution 
    function, and the field quantities with respect to time. This is used to evolve 
    the system with time.

    Input:
    ------

      Y0 : The array Y0 is the state of the system as given by the result of 
           the last time-step's integration. The elements of Y0, hold the following data:
     
           delta_f_hat   = Y0[0]
           delta_E_x_hat = Y0[1]
           delta_E_y_hat = Y0[2]
           delta_E_z_hat = Y0[3]
           delta_B_x_hat = Y0[4]
           delta_B_y_hat = Y0[5]
           delta_B_z_hat = Y0[6]
     
           At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    dY_dt : The time-derivatives of all the quantities stored in Y0
    """
    p1, p2, p3 = self.p1_center, self.p2_center, self.p3_center 

    k_x = self.k_x   
    k_y = self.k_y

    mass_particle   = self.mass_particle
    charge_electron = self.charge_electron

    delta_f_hat   = Y0[0]

    rhs = self._source_or_sink()

    ddelta_f_hat_dt = -1j * (k_x * vel_x + k_y * vel_y) * delta_f_hat + C_f
    
    dY_dt = np.array([ddelta_f_hat_dt])
    
    return(dY_dt)

  def _calculate_p(self):

    p1_center = self.p1_start  + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p2_center = self.p2_start  + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p3_center = self.p3_start  + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

    p2_center, p1_center, p3_center = np.meshgrid(p2_center, p1_center, p3_center)

    return(p1_center, p2_center, p3_center)

  def evolve(self, time_array, track_moments):
    # time_array needs to be specified including start time and the end time. 
    # Evaluating time-step size:
    dt = time_array[1] - time_array[0]

    if(len(track_moments) != 0):
      moments_data = np.zeros([time_array.size, len(track_moments)])

    for time_index, t0 in enumerate(time_array[1:]):
      print("Computing for Time =", t0)
      self.f = self._time_step(dt)

      for i in range(len(track_moments)):
        moments_data[time_index][i] = self.compute_moments(track_moments[i])

    return(moments_data)