#!/usr/bin/env python 

import arrayfire as af 
import numpy as np
from interpolation_routines import f_interp_2d 

from petsc4py import PETSc 

class nonlinear_solver(object):
  def __init__(self, physical_system):
    self.physical_system = physical_system
    self.log_f           = af.log(physical_system.f)

  def _convert(self, array):
    """
    This function is used to convert from velocities expanded
    form to positions expanded form and vice-versa.
    """
    
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self.physical_system.da.getCorners()

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

  _f_interp_2d = f_interp_2d

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