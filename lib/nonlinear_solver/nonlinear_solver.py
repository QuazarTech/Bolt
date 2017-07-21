#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

from lib.nonlinear_solver.communicate import communicate_distribution_function, communicate_fields

from lib.nonlinear_solver.timestepper import strang_step, lie_step
from lib.nonlinear_solver.compute_moments import compute_moments as compute_moments_imported
from lib.nonlinear_solver.EM_fields_solver.electrostatic import fft_poisson

class nonlinear_solver(object):
  def __init__(self, physical_system):
    self.physical_system = physical_system

    self.q1_start, self.q1_end = physical_system.q1_start, physical_system.q1_end
    self.q2_start, self.q2_end = physical_system.q2_start, physical_system.q2_end
    self.p1_start, self.p1_end = physical_system.p1_start, physical_system.p1_end
    self.p2_start, self.p2_end = physical_system.p2_start, physical_system.p2_end
    self.p3_start, self.p3_end = physical_system.p3_start, physical_system.p3_end

    self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1
    self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2
    self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1
    self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2
    self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3

    # Getting number of ghost zones, and the boundary conditions that are utilized
    self.N_ghost                 = physical_system.N_ghost
    self.bc_in_q1, self.bc_in_q2 = physical_system.bc_in_q1, physical_system.bc_in_q2

    # Declaring the communicator:
    self._comm = PETSc.COMM_WORLD.tompi4py()

    # The DA structure is used in domain decomposition:
    # The following DA is used in the communication routines where information
    # about the data of the distribution function needs to be communicated
    # amongst processes. Additionally this structure automatically
    # takes care of applying periodic boundary conditions.
    self._da = PETSc.DMDA().create([self.N_q1, self.N_q2],\
                                   dof = (self.N_p1 * self.N_p2 * self.N_p3),\
                                   stencil_width = self.N_ghost,\
                                   boundary_type = (self.bc_in_q1, self.bc_in_q2),\
                                   proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                                   stencil_type = 1, \
                                   comm = self._comm
                                  )

    self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],\
                                           dof = 6,\
                                           stencil_width = self.N_ghost,\
                                           boundary_type = (self.bc_in_q1, self.bc_in_q2),\
                                           proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                                           stencil_type = 1, \
                                           comm = self._comm
                                          )

    # Creation of the local and global vectors from the DA:
    self._glob  = self._da.createGlobalVec()
    self._local = self._da.createLocalVec()

    self._glob_fields  = self._da_fields.createGlobalVec()
    self._local_fields = self._da_fields.createLocalVec()

    # Obtaining the array values of the cannonical variables:
    self.q1_center, self.q2_center = self._calculate_q_center()

    self.p1, self.p2, self.p3 = self._calculate_p_center()

    # Assigning the function object to a method of nonlinear solver:
    self._initialize(physical_system.params)

    # Assigning the advection terms along q1 and q2
    self._A_q1 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[0]
    self._A_q2 = self.physical_system.A_q(self.p1, self.p2, self.p3, physical_system.params)[1]

    # Assigning the function objects to methods of the solver:
    self._A_p = self.physical_system.A_p

    self._source_or_sink = self.physical_system.source_or_sink
  
  def _convert(self, array):
    """
    This function is used to convert from velocities expanded
    form to positions expanded form and vice-versa.
    """
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    # Checking if in positionsExpanded form:
    if(array.shape[0] == N_q1_local + 2 * self.N_ghost):

      array  = af.moddims(array,
                          (N_q1_local + 2 * self.N_ghost)*\
                          (N_q2_local + 2 * self.N_ghost),\
                          self.N_p1,\
                          self.N_p2,\
                          self.N_p3
                         )
    else:

      array  = af.moddims(array,
                          (N_q1_local + 2 * self.N_ghost),\
                          (N_q2_local + 2 * self.N_ghost),\
                          self.N_p1*\
                          self.N_p2*\
                          self.N_p3,\
                          1
                         )

    af.eval(array)
    return(array)

  def _calculate_q_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    i_q1_center = i_q1_lowest + 0.5
    i_q2_center = i_q2_lowest + 0.5

    i_q1 = i_q1_center + np.arange(-self.N_ghost, N_q1_local + self.N_ghost)
    i_q2 = i_q2_center + np.arange(-self.N_ghost, N_q2_local + self.N_ghost)

    q1_center = af.to_array(self.q1_start  + i_q1 * self.dq1)
    q2_center = af.to_array(self.q2_start  + i_q2 * self.dq2)

    # Tiling such that variation in q1 is along axis 0:
    q1_center = af.tile(q1_center, 1, N_q2_local + 2*self.N_ghost,\
                        self.N_p1 * self.N_p2 * self.N_p3
                       )

    # Tiling such that variation in q2 is along axis 1:
    q2_center = af.tile(af.reorder(q2_center), N_q1_local + 2*self.N_ghost, 1,\
                        self.N_p1 * self.N_p2 * self.N_p3, 1
                       )

    af.eval(q1_center, q2_center)

    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(q1_center, q2_center)

  def _calculate_p_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    N_ghost = self.N_ghost

    p1_center = self.p1_start  + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p2_center = self.p2_start  + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p3_center = self.p3_start  + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

    p2_center, p1_center, p3_center = np.meshgrid(p2_center, p1_center, p3_center)
    
    p1_center = af.flat(af.to_array(p1_center))
    p2_center = af.flat(af.to_array(p2_center))
    p3_center = af.flat(af.to_array(p3_center))

    p1_center = af.tile(af.reorder(p1_center, 2, 3, 0, 1),\
                        N_q1_local + 2 * N_ghost, N_q2_local + 2 * N_ghost,\
                        1, 1
                       )

    p2_center = af.tile(af.reorder(p2_center, 2, 3, 0, 1),\
                        N_q1_local + 2 * N_ghost, N_q2_local + 2 * N_ghost,\
                        1, 1
                       )
    
    p3_center = af.tile(af.reorder(p3_center, 2, 3, 0, 1),\
                        N_q1_local + 2 * N_ghost, N_q2_local + 2 * N_ghost,\
                        1, 1
                       )

    af.eval(p1_center, p2_center, p3_center)
    # returned in positionsExpanded form
    # (N_q1, N_q2, N_p1*N_p2*N_p3)
    return(p1_center, p2_center, p3_center)

  def _initialize(self, params):
    self.f = self.physical_system.initial_conditions.\
             initialize_f(self.q1_center, self.q2_center,\
                          self.p1, self.p2, self.p3, params
                         )
    
    N_g = self.N_ghost

    self.normalization_constant = af.sum(self.f[N_g:-N_g, N_g:-N_g]) * \
                                  self.dp1 * self.dp2 * self.dp3/\
                                  (self.N_q1 * self.N_q2)
    self.f                      = self.f/self.normalization_constant

    self.physical_system.params.normalization_constant = self.normalization_constant
    return

    if(self.physical_system.params.fields_initialize == 'electrostatic'):
      fft_poisson(self)

      self.E1 = af.constant(0, self.E1.shape[0], self.E1.shape[1], dtype = af.Dtype.f64)
      
      self.B1 = af.constant(0, self.E1.shape[0], self.E1.shape[1], dtype = af.Dtype.f64)
      self.B2 = af.constant(0, self.E1.shape[0], self.E1.shape[1], dtype = af.Dtype.f64)
      self.B3 = af.constant(0, self.E1.shape[0], self.E1.shape[1], dtype = af.Dtype.f64)

  # Injection of solver functions into class as methods:
  _communicate_distribution_function = communicate_distribution_function
  _communicate_fields = communicate_fields

  strang_timestep = strang_step
  lie_timestep    = lie_step

  compute_moments = compute_moments_imported