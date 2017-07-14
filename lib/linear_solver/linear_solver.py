#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2, ifft2, fftfreq

from lib.linear_solver.timestepper import RK6_step

class linear_solver(object):
  def __init__(self, physical_system):
    self.physical_system = physical_system

    # Storing domain information from physical system:
    # Getting resolution and size of configuration and velocity space:
    self.N_q1, self.q1_start, self.q1_end = physical_system.N_q1, physical_system.q1_start, physical_system.q1_end
    self.N_q2, self.q2_start, self.q2_end = physical_system.N_q2, physical_system.q2_start, physical_system.q2_end
    self.N_p1, self.p1_start, self.p1_end = physical_system.N_p1, physical_system.p1_start, physical_system.p1_end
    self.N_p2, self.p2_start, self.p2_end = physical_system.N_p2, physical_system.p2_start, physical_system.p2_end
    self.N_p3, self.p3_start, self.p3_end = physical_system.N_p3, physical_system.p3_start, physical_system.p3_end

    # Evaluating step size:
    self.dq1 = physical_system.dq1 
    self.dq2 = physical_system.dq2
    self.dp1 = physical_system.dp1
    self.dp2 = physical_system.dp2
    self.dp3 = physical_system.dp3

    # Getting number of ghost zones, and the boundary conditions that are utilized
    self.N_ghost               = physical_system.N_ghost
    self.bc_in_x, self.bc_in_y = physical_system.bc_in_x, physical_system.bc_in_y 

    if(self.bc_in_x != 'periodic' or self.bc_in_y != 'periodic'):
      raise Exception('Only systems with periodic boundary conditions can be solved using the linear solver')

    self.q1_center, self.q2_center = self._calculate_q_center()
    self.k_q1,      self.k_q2      = self._calculate_k()

    self.p1, self.p2, self.p3      = self._calculate_p()

    self._A_q1 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[0]
    self._A_q2 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[1]

    # Initializing f and f_hat:
    self._init(physical_system.params)

    # Assigning the function objects to methods of the solver:
    self.A_p = self.physical_system.A_p

    self._source_or_sink = self.physical_system.source_or_sink

  def _calculate_q_center(self):
    q1_center = self.q1_start + (0.5 + np.arange(self.N_q1)) * self.dq1
    q2_center = self.q2_start + (0.5 + np.arange(self.N_q2)) * self.dq2

    q2_center, q1_center = np.meshgrid(q2_center, q1_center)
    
    q2_center = q2_center.reshape(self.N_q1, self.N_q2, 1, 1, 1)
    q1_center = q1_center.reshape(self.N_q1, self.N_q2, 1, 1, 1)

    q2_center = np.tile(q2_center, (1, 1, self.N_p1, self.N_p2, self.N_p3))
    q1_center = np.tile(q1_center, (1, 1, self.N_p1, self.N_p2, self.N_p3))

    return(q1_center, q2_center)

  def _calculate_k(self):
    k_q1 = 2 * np.pi * fftfreq(self.N_q1, self.dq1)
    k_q2 = 2 * np.pi * fftfreq(self.N_q2, self.dq2)

    k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

    k_q2 = k_q2.reshape(self.N_q1, self.N_q2, 1, 1, 1)
    k_q1 = k_q1.reshape(self.N_q1, self.N_q2, 1, 1, 1)

    k_q2 = np.tile(k_q2, (1, 1, self.N_p1, self.N_p2, self.N_p3))
    k_q1 = np.tile(k_q1, (1, 1, self.N_p1, self.N_p2, self.N_p3))

    return(k_q1, k_q2)

  def _calculate_p(self):

    p1_center = self.p1_start  + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p2_center = self.p2_start  + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p3_center = self.p3_start  + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

    p2_center, p1_center, p3_center = np.meshgrid(p2_center, p1_center, p3_center)
    
    p1_center = p1_center.reshape(1, 1, self.N_p1, self.N_p2, self.N_p3)
    p2_center = p2_center.reshape(1, 1, self.N_p1, self.N_p2, self.N_p3)
    p3_center = p3_center.reshape(1, 1, self.N_p1, self.N_p2, self.N_p3)

    p1_center = np.tile(p1_center, (self.N_q1, self.N_q2, 1, 1, 1))
    p2_center = np.tile(p2_center, (self.N_q1, self.N_q2, 1, 1, 1))
    p3_center = np.tile(p3_center, (self.N_q1, self.N_q2, 1, 1, 1))

    return(p1_center, p2_center, p3_center)

  def _df_dv_background(self):

    if(self.physical_system.p_dim == 1):
      dfdp1_background = np.gradient(self.f_background[:, 0, 0], self.dp1)
      dfdp2_background = np.zeros_like(self.f_background)
      dfdp3_background = np.zeros_like(self.f_background)

    elif(self.physical_system.p_dim == 2):
      dfdp1_background = np.gradient(self.f_background[:, :, 0], self.dp1, self.dp2)[0]
      dfdp2_background = np.gradient(self.f_background[:, :, 0], self.dp1, self.dp2)[1]
      dfdp3_background = np.zeros_like(self.f_background)
    
    else:
      dfdp1_background = np.gradient(self.f_background, self.dp1, self.dp2, self.dp3)[0]
      dfdp2_background = np.gradient(self.f_background, self.dp1, self.dp2, self.dp3)[1]
      dfdp3_background = np.gradient(self.f_background, self.dp1, self.dp2, self.dp3)[2]

    self.dfdp1_background = dfdp1_background.reshape([1, 1, self.dp1, self.dp2, self.dp3])
    self.dfdp2_background = dfdp2_background.reshape([1, 1, self.dp1, self.dp2, self.dp3])
    self.dfdp2_background = dfdp3_background.reshape([1, 1, self.dp1, self.dp2, self.dp3])

    return

  def _init(self, params):
    f           = self.physical_system.initial_conditions(self.q1_center, self.q2_center,\
                                                          self.p1, self.p2, self.p3, params
                                                         )
    self.f_hat  = fft2(f, axes = (0, 1))

    self.f_background           = abs(self.f_hat[0, 0, :, :, :])/(self.N_q1 * self.N_q2)
    self.normalization_constant = np.sum(self.f_background) * self.dp1 * self.dp2 * self.dp3
    
    self.f_background = self.f_background/self.normalization_constant
    self.f_hat        = self.f_hat/self.normalization_constant
    
    self.dfdp1_background, self.dfdp2_background, self.dfdp3_background = \
    self._df_dv_background()
    
    # Scaling Appropriately:
    self.f_hat = 2*self.f_hat/(self.N_q1 * self.N_q2)
    
    # Appending the normalization constant to params:
    self.physical_system.params.normalization_constant = self.normalization_constant

    self.Y = np.array([self.f_hat])
    return

  def compute_moments(self, moment_name):
    try:
      moment_exponents = np.array(self.physical_system.moment_exponents[moment_name])
      moment_coeffs    = np.array(self.physical_system.moment_coeffs[moment_name])

    except:
      raise KeyError('moment_name not defined under physical system')

    try:
      moment_variable = 1
      for i in range(moment_exponents.shape[0]):
        moment_variable *= moment_coeffs[i, 0] * self.p1**(moment_exponents[i, 0]) + \
                           moment_coeffs[i, 1] * self.p2**(moment_exponents[i, 1]) + \
                           moment_coeffs[i, 2] * self.p3**(moment_exponents[i, 2])
    except:
      moment_variable = moment_coeffs[0] * self.p1**(moment_exponents[0]) + \
                        moment_coeffs[1] * self.p2**(moment_exponents[1]) + \
                        moment_coeffs[2] * self.p3**(moment_exponents[2])

    moment_hat = np.sum(np.sum(np.sum(self.Y[0] * moment_variable, 4)*self.dp3, 3)*self.dp2, 2)*self.dp1

    # Scaling Appropriately:
    moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
    moment     = ifft2(moment_hat).real
    return(moment)

  def _dY_dt(self, Y):
    """
    Returns the value of the derivative of the mode perturbation of the distribution 
    function, and the field quantities with respect to time. This is used to evolve 
    the system with time.

    Input:
    ------

      Y0 : The array Y is the state of the system as given by the result of 
           the last time-step's integration. The elements of Y, hold the following data:
     
           delta_f_hat = Y[0]
     
           At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    dY_dt : The time-derivatives of all the quantities stored in Y
    """
    self.f_hat   = Y[0]
    self.E_x_hat = Y[1]
    self.E_y_hat = Y[2]
    self.E_z_hat = Y[3]
    self.B_x_hat = Y[4]
    self.B_y_hat = Y[5]
    self.B_z_hat = Y[6]
    
    # Scaling Appropriately:
    self.f     = ifft2(0.5 * self.N_q2 * self.N_q1 * self.f_hat, axes = (0, 1))
    C_f_hat    = 2 * fft2(self._source_or_sink(self.f, self.q1_center, self.q2_center,\
                                               self.p1, self.p2, self.p3,\
                                               self.compute_moments, self.physical_system.params
                                              ),axes = (0, 1))/(self.N_q2 * self.N_q1)
    
    mom_bulk_p1 = self.compute_moments('mom_p1_bulk')
    mom_bulk_p2 = self.compute_moments('mom_p2_bulk')
    mom_bulk_p3 = self.compute_moments('mom_p3_bulk')

    charge_electron = self.physical_system.params.charge_electron
    mass_particle   = self.physical_system.params.mass_particle 

    J1_hat = self.physical_system.params.charge_electron * mom_bulk_p1
    J2_hat = self.physical_system.params.charge_electron * mom_bulk_p2
    J3_hat = self.physical_system.params.charge_electron * mom_bulk_p3
    
    dE1_hat_dt = (self.B3_hat * 1j * self.k_q2) - J1_hat
    dE2_hat_dt = (- self.B3_hat * 1j * self.k_q1) - J2_hat
    dE3_hat_dt = (self.B2_hat * 1j * self.k_q1 - self.B1_hat * 1j * self.k_q2) - J3_hat

    dB1_hat_dt = (- self.E3_hat * 1j * self.k_q2)
    dB2_hat_dt = (self.E3_hat * 1j * self.k_q1)
    dB3_hat_dt = (self.E1_hat * 1j * self.k_q2 - self.E1_hat * 1j * self.k_q1)

    fields_term = (charge_electron / mass_particle) * (self.E1_hat + \
                                                       self.B3_hat * self.p2 - \
                                                       self.B2_hat * self.p3
                                                      ) * self.dfdp1_background  + \
                  (charge_electron / mass_particle) * (self.E1_hat + \
                                                       self.B1_hat * self.p3 - \
                                                       self.B3_hat * self.p1
                                                      ) * self.dfdp2_background  + \
                  (charge_electron / mass_particle) * (self.E3_hat + \
                                                       self.B2_hat * self.p1 -\
                                                       self.B1_hat * self.p2
                                                      ) * self.dfdp3_background

    df_hat_dt  = -1j * (self.k_q1 * self._A_q1 + self.k_q2 * self._A_q2) * self.f_hat + \
                 C_f_hat - fields_term
    
    dY_dt = np.array([df_hat_dt,\
                      dE1_hat_dt, dE2_hat_dt, dE3_hat_dt,\
                      dB1_hat_dt, dB2_hat_dt, dB3_hat_dt,\
                     ]
                    )

    return(dY_dt)

  time_step = RK6_step