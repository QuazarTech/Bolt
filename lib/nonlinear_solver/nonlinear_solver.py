#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af 
import numpy as np
from petsc4py import PETSc 
import h5py

# Importing solver functions:
from interpolation_routines import f_interp_2d
from solve_source_sink import RK2 

class nonlinear_solver(object):
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
    
    # Creation of the local and global vectors from the DA:
    self._glob  = self._da.createGlobalVec()
    self._local = self._da.createLocalVec()

    # Obtaining the array values of the cannonical variables: 
    self.q1_center = self._calculate_q1_center()
    self.q2_center = self._calculate_q2_center()
    self.p1_center = self._calculate_p1_center()
    self.p2_center = self._calculate_p2_center()
    self.p3_center = self._calculate_p3_center()

    # Initializing the distribution function(s):
    self.physical_system.init(*args, **kwargs)

    self._A_q1 = self.physical_system.A_q1(self.p1_center, self.p2_center, self.p3_center)
    self._A_q2 = self.physical_system.A_q2(self.p1_center, self.p2_center, self.p3_center)
    self._A_p1 = self.physical_system.A_p1()
    self._A_p2 = self.physical_system.A_p2()
    self._A_p3 = self.physical_system.A_p3()

    self._source_or_sink = self.physical_system.source_or_sink
  
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
    local_value = self.physical_system.da.getVecArray(self._local)
    glob_value  = self.physical_system.da.getVecArray(self._glob)

    N_ghost = self.physical_system.N_ghost

    # Storing values of af.Array in PETSc.Vec:
    local_value[:] = np.array(self.f)
    
    # Global value is non-inclusive of the ghost-zones:
    glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                     N_ghost:-N_ghost,\
                                     :
                                    ]

    # The following function takes care of periodic boundary conditions, 
    # and interzonal communications:
    self.physical_system.da.globalToLocal(self.glob, self.local)

    # Converting back from PETSc.Vec to af.Array:
    self.f = af.to_array(local_value[:])

    af.eval(self.f)
    return

  # Injection of solver functions into class as methods:
  _f_interp_2d       = f_interp_2d
  _solve_source_sink = RK2

  def _time_step(self, dt):
    # Advection in position space:
    self._f_interp_2d(self, 0.25*dt)
    self._communicate_distribution_function()
    # Advection in position space:
    self._f_interp_2d(self, 0.25*dt)
    self._communicate_distribution_function()
    # Advection in position space:
    self._f_interp_2d(self, 0.25*dt)
    self._communicate_distribution_function()
    # Advection in position space:
    self._f_interp_2d(self, 0.25*dt)
    self._communicate_distribution_function()

    af.eval(self.f)
    return(self.f)

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

    moment = af.sum(af.sum(af.sum(self.f * moment_variable, 3)*self.dp3, 2)*self.dp2, 1)*self.dp1
    
    af.eval(moment)
    return(moment)

  def evolve(self, time_array, track_moments):
    # time_array needs to be specified including start time and the end time. 
    # Evaluating time-step size:
    dt = time_array[1] - time_array[0]

    if(len(track_moments) != 0):
      moments_data = np.zeros([time_array.size, len(track_moments)])

    for time_index, t0 in enumerate(time_array[1:]):
      PETSc.Sys.Print("Computing for Time =", t0)
      self.f = self._time_step(dt)

      for i in range(len(track_moments)):
        moments_data[time_index][i] = self.compute_moments(track_moments[i])

    return(moments_data)

  def dump_variables(self, file_name, **args):
    h5f = h5py.File(file_name + '.h5', 'w')
    for variable_name in args:
      h5f.create_dataset(str(variable_name), data = variable_name)
    h5f.close()
    return

  def dump_distribution_function(self, file_name):
    PETSc.Object.setName(self._glob, 'distribution_function')
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm = self._comm)
    
    global_vec_value    = self._da.getVecArray(self._glob)
    global_vec_value[:] = np.array(self.f[self.N_ghost:-self.N_ghost,\
                                          self.N_ghost:-self.N_ghost, :]
                                  )
    viewer(self._glob)
    return