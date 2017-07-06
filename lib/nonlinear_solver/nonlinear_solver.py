#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af 
import numpy as np
from interpolation_routines import f_interp_2d 

from petsc4py import PETSc 

class nonlinear_solver(object):
  def __init__(self, physical_system):
    self.physical_system = physical_system
    self.log_f           = af.log(physical_system.f)

    # Declaring the communicator:
    self._comm = PETSc.COMM_WORLD.tompi4py()

    # The DA structure is used in domain decomposition:
    # The following DA is used in the communication routines where information 
    # about the data of the distribution function needs to be communicated 
    # amongst processes. Additionally this structure automatically
    # takes care of applying periodic boundary conditions.
    self._da = PETSc.DMDA().create([physical_system.N_q1, physical_system.N_q2],\
                                   dof = (physical_system.N_p1 * physical_system.N_p2 * physical_system.N_p3),\
                                   stencil_width = physical_system.N_ghost,\
                                   boundary_type = (physical_system.bc_in_x, physical_system.bc_in_y),\
                                   proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                                   stencil_type = 1, \
                                   comm = self._comm
                                  )
    
    # Additionally, We'll define another DA so that communication of electromagnetic
    # field quantities may be performed, in addition to applying periodic B.C's
    # We define this DA with a DOF of 6 so that the communication for all field
    # quantities(Ex, Ey, Ez, Bx, By, Bz) can be carried out with a single call to
    # the communication routine
    self._da_fields = PETSc.DMDA().create([physical_system.N_q1, physical_system.N_q2],\
                                           dof = 6,\
                                           stencil_width = physical_system.N_ghost,\
                                           boundary_type = self._da.getBoundaryType(),\
                                           proc_sizes = self._da.getProcSizes(), \
                                           stencil_type = 1, \
                                           comm = self._da.getComm()
                                          )
    
    # Obtaining the array values of the cannonical variables: 
    self.q1_center = self._calculate_q1_center()
    self.q2_center = self._calculate_q2_center()
    self.p1_center = self._calculate_p1_center()
    self.p2_center = self._calculate_p2_center()
    self.p3_center = self._calculate_p3_center()

  def _calculate_q1_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    i_center = i_q1_lowest + 0.5
    i        = i_center + np.arange(-self.N_ghost, N_q1_local + self.N_ghost, 1)
    
    q1_center = self.q1_start  + i * self.dq1
    q1_center = af.Array.as_type(af.to_array(q1_center), af.Dtype.f64)

    # Tiling such that variation in q1 is along axis 0:
    q1_center = af.tile(q1_center, 1, N_q2_local + 2*self.N_ghost,\
                        self.N_p1 * self.N_p2 * self.N_p3, 1
                       )

    af.eval(q1_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(q1_center)

  def _calculate_q2_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    i_center = i_q2_lowest + 0.5
    i        = i_center + np.arange(-self.N_ghost, N_q2_local + self.N_ghost, 1)
    
    q2_center = self.q2_start  + i * self.dq2
    q2_center = af.Array.as_type(af.to_array(q2_center), af.Dtype.f64)

    # Tiling such that variation in q2 is along axis 1:
    q2_center = af.tile(af.reorder(q2_center), N_q1_local + 2*self.N_ghost, 1,\
                        self.N_p1 * self.N_p2 * self.N_p3, 1
                       )

    af.eval(q2_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(q2_center)

  def _calculate_p1_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    p1_center = self.p1_start  + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p1_center = af.Array.as_type(af.to_array(p1_center), af.Dtype.f64)

    # Tiling such that variation in p1 is along axis 1:
    p1_center = af.tile(af.reorder(p1_center), (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        1, self.N_p2, self.N_p3
                       )

    p1_center = af.moddims(p1_center,                   
                           (N_q1_local + 2 * self.N_ghost),\
                           (N_q2_local + 2 * self.N_ghost),\
                           self.N_p1*\
                           self.N_p2*\
                           self.N_p3,\
                           1
                          )

    af.eval(p1_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(p1_center)

  def _calculate_p2_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    p2_center = self.p2_start  + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p2_center = af.Array.as_type(af.to_array(p2_center), af.Dtype.f64)

    # Tiling such that variation in p2 is along axis 2:
    p2_center = af.tile(af.reorder(p2_center, 2, 3, 0, 1),\
                        (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        self.N_p1, 1, self.N_p3
                       )

    p2_center = af.moddims(p2_center,                   
                           (N_q1_local + 2 * self.N_ghost),\
                           (N_q2_local + 2 * self.N_ghost),\
                           self.N_p1*\
                           self.N_p2*\
                           self.N_p3,\
                           1
                          )

    af.eval(p2_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(p2_center)

  def _calculate_p3_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    p3_center = self.p3_start  + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3
    p3_center = af.Array.as_type(af.to_array(p3_center), af.Dtype.f64)

    # Tiling such that variation in p3 is along axis 3:
    p3_center = af.tile(af.reorder(p3_center, 1, 2, 3, 0),\
                        (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        self.N_p1, self.N_p2, 1
                       )

    p3_center = af.moddims(p3_center,                   
                           (N_q1_local + 2 * self.N_ghost),\
                           (N_q2_local + 2 * self.N_ghost),\
                           self.N_p1*\
                           self.N_p2*\
                           self.N_p3,\
                           1
                          )
    af.eval(p3_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(p3_center)

  def _convert(self, array):
    """
    This function is used to convert from velocities expanded
    form to positions expanded form and vice-versa.
    """
    
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self.physical_system._da.getCorners()

    # Checking if in positionsExpanded form:
    if(array.shape[0] == self.physical_system.N_q1 + 2 * self.physical_system.N_ghost):

      array  = af.moddims(array,                   
                          (N_q1_local + 2 * self.physical_system.N_ghost)*\
                          (N_q2_local + 2 * self.physical_system.N_ghost),\
                          self.physical_system.N_p1,\
                          self.physical_system.N_p2,\
                          self.physical_system.N_p3
                         ) 
    else:

      array  = af.moddims(array,                   
                          (N_q1_local + 2 * self.physical_system.N_ghost),\
                          (N_q2_local + 2 * self.physical_system.N_ghost),\
                          self.physical_system.N_p1*\
                          self.physical_system.N_p2*\
                          self.physical_system.N_p3,\
                          1
                         )
    
    af.eval(array)
    return(array) 

  def _communicate_distribution_function(self):

    # Accessing the values of the global and local Vectors
    local_value = self.physical_system.da.getVecArray(self.local)
    glob_value  = self.physical_system.da.getVecArray(self.glob)

    N_ghost = self.physical_system.N_ghost

    # Storing values of af.Array in PETSc.Vec:
    local_value[:] = np.array(self.log_f)
    
    # Global value is non-inclusive of the ghost-zones:
    glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                     N_ghost:-N_ghost,\
                                     :
                                    ]

    # The following function takes care of periodic boundary conditions, 
    # and interzonal communications:
    self.physical_system.da.globalToLocal(self.glob, self.local)

    # Converting back from PETSc.Vec to af.Array:
    self.log_f = af.to_array(local_value[:])

    af.eval(self.log_f)
    return


  # Injection of functions into class:
  _f_interp_2d       = f_interp_2d
  _solve_source_sink = solve_source_sink

  def evolve(self, time_array):
    # time_array needs to be specified including start time and the end time. 

    # Creation of the local and global vectors from the DA:
    self.glob  = self.physical_system.da.createGlobalVec()
    self.local = self.physical_system.da.createLocalVec()

    for time_index, t0 in enumerate(time_array[1:]):
      PETSc.Sys.Print("Computing for Time =", t0)

      dt = time_array[1] - time_array[0]

      # Advection in position space:
      self.log_f = self._f_interp_2d(self, 0.25*dt)
      self._communicate_distribution_function()
      # Advection in position space:
      self.log_f = self._f_interp_2d(self, 0.25*dt)
      self._communicate_distribution_function()
      # Advection in position space:
      self.log_f = self._f_interp_2d(self, 0.25*dt)
      self._communicate_distribution_function()
      # Advection in position space:
      self.log_f = self._f_interp_2d(self, 0.25*dt)
      self._communicate_distribution_function()
    
    self.glob.destroy()
    self.local.destroy()

    return

  def dump(self, file_name, **args):
    h5f = h5py.File(file_name + '.h5', 'w')
    for variable_name in args:
      h5f.create_dataset(str(variable_name), data = variable_name)
    h5f.close()

  def compute_moments(self, moment_variable):
    moment = af.sum(af.sum(af.sum(self.f * moment_variable, 3)*self.dp3, 2)*self.dp2, 1)*self.dp1
    
    af.eval(moment)
    return(moment)