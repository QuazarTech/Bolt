#!/usr/bin/env python 

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
    self._da = PETSc.DMDA().create([self.N_q1, self.N_q2],\
                                   dof = (self.N_p1 * self.N_p2 * self.N_p3),\
                                   stencil_width = self.N_ghost,\
                                   boundary_type = (self.bc_in_x, self.bc_in_y),\
                                   proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                                   stencil_type = 1, \
                                   comm = self._comm
                                  )

    # Obtaining the array values of the cannonical variables: 
    self.q1_center = self._calculate_q1_center()
    self.q2_center = self._calculate_q2_center()
    self.p1_center = self._calculate_p1_center()
    self.p2_center = self._calculate_p2_center()
    self.p3_center = self._calculate_p3_center()

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

  def dump(self, file_name, **args):
    h5f = h5py.File(file_name + '.h5', 'w')
    for variable_name in args:
      h5f.create_dataset(str(variable_name), data = variable_name)
    h5f.close()

  def compute_moments(self, moment_variable):
    moment = af.sum(af.sum(af.sum(self.f * moment_variable, 3)*self.dp3, 2)*self.dp2, 1)*self.dp1
    
    af.eval(moment)
    return(moment)