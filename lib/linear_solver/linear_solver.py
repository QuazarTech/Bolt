#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import h5py

class linear_system(object):
  def __init__(self, physical_system):
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

    self.p1_center = self._calculate_p1_center()
    self.p2_center = self._calculate_p2_center()
    self.p3_center = self._calculate_p3_center()

    self._source_or_sink = self.physical_system.source_or_sink
  
  def compute_moments(self, moment_name):

    moment_exponents = np.array(self.physical_system.moments[moment_name])
    
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

  def dump_distribution_function_5D(self, file_name):

    return

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