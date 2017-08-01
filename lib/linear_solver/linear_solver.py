#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

# In this code, we shall default to using the positionsExpanded form thoroughout.
# This means that the arrays defined in the system will be of the form:
# (N_q1, N_q2, N_p1*N_p2*N_p3)

# Importing dependencies:
import numpy as np
import arrayfire as af
from scipy.fftpack import fftfreq

# Importing solver functions:
from lib.linear_solver.dY_dt import dY_dt

from lib.linear_solver.timestepper import RK2_step as RK2_step_imported
from lib.linear_solver.timestepper import RK4_step as RK4_step_imported
from lib.linear_solver.timestepper import RK6_step as RK6_step_imported

from lib.linear_solver.EM_fields_solver import compute_electrostatic_fields
from lib.linear_solver.calculate_dfdp_background import calculate_dfdp_background
from lib.linear_solver.compute_moments import compute_moments as compute_moments_imported

import lib.linear_solver.dump as dump

class linear_solver(object):
  """
  An instance of this class' attributes contains methods which are used in evolving
  the system declared under physical system. The state of the system then may be 
  determined from the attributes of the system such as the distribution function and
  electromagnetic fields
  """
  def __init__(self, physical_system):
    """
    Constructor for the linear_solver object. It takes the physical 
    system object as an argument and uses it in intialization and 
    evolution of the system in consideration.
    """
    self.physical_system = physical_system

    # Storing Domain Information:
    self.q1_start, self.q1_end = physical_system.q1_start, physical_system.q1_end
    self.q2_start, self.q2_end = physical_system.q2_start, physical_system.q2_end
    self.p1_start, self.p1_end = physical_system.p1_start, physical_system.p1_end
    self.p2_start, self.p2_end = physical_system.p2_start, physical_system.p2_end
    self.p3_start, self.p3_end = physical_system.p3_start, physical_system.p3_end
     
    # Getting Domain Resolution
    self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1 
    self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2 
    self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1 
    self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2 
    self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3 

    # Getting number of ghost zones, and the boundary conditions that are utilized
    self.N_ghost                 = physical_system.N_ghost
    self.bc_in_q1, self.bc_in_q2 = physical_system.bc_in_q1, physical_system.bc_in_q2

    # Checking that periodic B.C's are utilized:
    if(self.bc_in_q1 != 'periodic' or self.bc_in_q2 != 'periodic'):
      raise Exception('Only systems with periodic boundary conditions can be solved using the linear solver')

    # Intializing position, wavenumber and velocity arrays:
    self.q1_center, self.q2_center = self._calculate_q_center()
    self.k_q1,      self.k_q2      = self._calculate_k()

    self.p1, self.p2, self.p3      = self._calculate_p()

    # Assigning the advection terms along q1 and q2
    self._A_q1 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[0]
    self._A_q2 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[1]

    # Initializing f, f_hat and the other EM field quantities:
    self._initialize(physical_system.params)

    # Assigning the function objects to methods of the solver:
    self._A_p = self.physical_system.A_p

    self._source_or_sink = self.physical_system.source_or_sink

  def _calculate_q_center(self):
    """
    Initializes the cannonical variables q1, q2 using a centered
    formulation. The size, and resolution are the same as declared 
    under domain of the physical system object.
    """
    q1_center = self.q1_start + (0.5 + np.arange(self.N_q1)) * self.dq1
    q2_center = self.q2_start + (0.5 + np.arange(self.N_q2)) * self.dq2

    q2_center, q1_center = np.meshgrid(q2_center, q1_center)
    
    q2_center = af.to_array(q2_center)
    q1_center = af.to_array(q1_center)

    q2_center = af.tile(q2_center, 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
    q1_center = af.tile(q1_center, 1, 1, self.N_p1 * self.N_p2 * self.N_p3)

    af.eval(q1_center, q2_center)
    
    # returned in positionsExpanded form
    # (N_q1, N_q2, N_p1*N_p2*N_p3)
    return(q1_center, q2_center)

  def _calculate_k(self):
    """
    Initializes the wave numbers k_q1 and k_q2 which will be used when
    solving in fourier space.
    """
    k_q1 = 2 * np.pi * fftfreq(self.N_q1, self.dq1)
    k_q2 = 2 * np.pi * fftfreq(self.N_q2, self.dq2)

    k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

    k_q2 = af.to_array(k_q2)
    k_q1 = af.to_array(k_q1)

    k_q2 = af.tile(k_q2, 1, 1, self.N_p1*self.N_p2*self.N_p3)
    k_q1 = af.tile(k_q1, 1, 1, self.N_p1*self.N_p2*self.N_p3)

    af.eval(k_q1, k_q2)
    
    # returned in positionsExpanded form
    # (N_q1, N_q2, N_p1*N_p2*N_p3)
    return(k_q1, k_q2)

  def _calculate_p(self):
    """
    Initializes the cannonical variables p1, p2 and p3 using a centered
    formulation. The size, and resolution are the same as declared 
    under domain of the physical system object.
    """
    p1_center = self.p1_start  + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p2_center = self.p2_start  + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p3_center = self.p3_start  + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

    p2_center, p1_center, p3_center = np.meshgrid(p2_center, p1_center, p3_center)
    
    p1_center = af.flat(af.to_array(p1_center))
    p2_center = af.flat(af.to_array(p2_center))
    p3_center = af.flat(af.to_array(p3_center))

    p1_center = af.tile(af.reorder(p1_center, 2, 3, 0, 1), self.N_q1, self.N_q2, 1, 1)
    p2_center = af.tile(af.reorder(p2_center, 2, 3, 0, 1), self.N_q1, self.N_q2, 1, 1)
    p3_center = af.tile(af.reorder(p3_center, 2, 3, 0, 1), self.N_q1, self.N_q2, 1, 1)

    af.eval(p1_center, p2_center, p3_center)
    # returned in positionsExpanded form
    # (N_q1, N_q2, N_p1*N_p2*N_p3)
    return(p1_center, p2_center, p3_center)

  # Assigning function that is used in computiong the derivatives
  # of the background distribution function:
  _calculate_dfdp_background = calculate_dfdp_background

  def _initialize(self, params):
    """
    Called when the solver object is declared. This function is
    used to initialize the mode perturbation of the distribution 
    function f_hat, along with the mode perturbation of the field quantities
    """
    f           = self.physical_system.initial_conditions.\
                  initialize_f(self.q1_center, self.q2_center,\
                               self.p1, self.p2, self.p3, params
                              )
    
    self.f_hat        = af.fft2(f)

    # Since (k_q1, k_q2) = (0, 0) will give the background distribution:
    self.f_background = af.abs(self.f_hat[0, 0, :])/(self.N_q1 * self.N_q2)
    
    # Calculating derivatives of the background distribution function:
    self._calculate_dfdp_background()
    
    # Scaling Appropriately:
    self.f_hat         = 2*self.f_hat/(self.N_q1 * self.N_q2)

    # Using a vector Y to evolve the system:
    self.Y             = af.constant(0, self.p1.shape[0], self.p1.shape[1],\
                                     self.p1.shape[2], 7, dtype = af.Dtype.c64
                                    )
    
    self.Y[:, :, :, 0] = self.f_hat
    
    # Initializing EM fields using Poisson Equation:
    if(self.physical_system.params.fields_initialize == 'electrostatic'):
      compute_electrostatic_fields(self)

    # If option is given as user-defined:
    elif(self.physical_system.params.fields_initialize == 'user-defined'):
      E1, E2, E3 = self.initial_conditions.initialize_E(self.physical_system.params)
      B1, B2, B3 = self.initial_conditions.initialize_B(self.physical_system.params)

      # Scaling Appropriately
      self.E1_hat = 2 * af.fft2(E1)/(self.N_q1 * self.N_q2)
      self.E2_hat = 2 * af.fft2(E2)/(self.N_q1 * self.N_q2)
      self.E3_hat = 2 * af.fft2(E3)/(self.N_q1 * self.N_q2)
      self.B1_hat = 2 * af.fft2(B1)/(self.N_q1 * self.N_q2)
      self.B2_hat = 2 * af.fft2(B2)/(self.N_q1 * self.N_q2)
      self.B3_hat = 2 * af.fft2(B3)/(self.N_q1 * self.N_q2)

    self.Y[:, :, :, 1] = self.E1_hat
    self.Y[:, :, :, 2] = self.E2_hat
    self.Y[:, :, :, 3] = self.E3_hat
    self.Y[:, :, :, 4] = self.B1_hat
    self.Y[:, :, :, 5] = self.B2_hat
    self.Y[:, :, :, 6] = self.B3_hat

    return

  # Injection of solver methods from other files:
  _dY_dt   = dY_dt

  RK2_step = RK2_step_imported
  RK4_step = RK4_step_imported
  RK6_step = RK6_step_imported

  compute_moments            = compute_moments_imported

  dump_distribution_function = dump.dump_distribution_function
  dump_variables             = dump.dump_variables