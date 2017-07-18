#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

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
    self.bc_in_q1, self.bc_in_q2 = physical_system.in_q1, physical_system.in_q2

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

    self._glob_field  = self._da_fields.createGlobalVec()
    self._local_field = self._da_fields.createLocalVec()

    # Obtaining the array values of the cannonical variables:
    self.q1_center = self._calculate_q1_center()
    self.q2_center = self._calculate_q2_center()

    self.p1, self.p2, self.p3 = self._calculate_p_center()

    # Assigning the function object to a method of nonlinear solver:
    self._init(physical_system.params)

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
    if(array.shape[0] == self.N_q1 + 2 * self.N_ghost):

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

  def _calculate_p_center(self):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

    p1_center = self.p1_start + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
    p1_center = af.Array.as_type(af.to_array(p1_center), af.Dtype.f64)
    p2_center = self.p2_start + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
    p2_center = af.Array.as_type(af.to_array(p2_center), af.Dtype.f64)
    p3_center = self.p3_start + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3
    p3_center = af.Array.as_type(af.to_array(p3_center), af.Dtype.f64)

    # Tiling such that variation in p1 is along axis 1:
    p1_center = af.tile(af.reorder(p1_center), (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        1, self.N_p2, self.N_p3
                       )
    # Tiling such that variation in p2 is along axis 2:
    p2_center = af.tile(af.reorder(p2_center, 2, 3, 0, 1),\
                        (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        self.N_p1, 1, self.N_p3
                       )
    # Tiling such that variation in p3 is along axis 3:
    p3_center = af.tile(af.reorder(p3_center, 1, 2, 3, 0),\
                        (N_q1_local + 2*self.N_ghost)*(N_q2_local + 2*self.N_ghost),\
                        self.N_p1, self.N_p2, 1
                       )

    # Converting from velocitiesExpanded form to positionsExpanded form:
    p1_center = self._convert(p1_center)
    p2_center = self._convert(p2_center)
    p3_center = self._convert(p3_center)

    af.eval(p1_center, p2_center, p3_center)
    # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
    return(p1_center, p2_center, p3_center)

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

  def _communicate_fields(self):

    # Accessing the values of the global and local Vectors
    local_value = self._da.getVecArray(self._local_field)
    glob_value  = self._da.getVecArray(self._glob_field)

    N_ghost = self.N_ghost

    # Assigning the values of the af.Array fields quantities
    # to the PETSc.Vec:
    (local_value[:])[:, :, 0] = np.array(self.E1)
    (local_value[:])[:, :, 1] = np.array(self.E2)
    (local_value[:])[:, :, 2] = np.array(self.E3)

    (local_value[:])[:, :, 3] = np.array(self.B1)
    (local_value[:])[:, :, 4] = np.array(self.B2)
    (local_value[:])[:, :, 5] = np.array(self.B3)

    # Global value is non-inclusive of the ghost-zones:
    glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                     N_ghost:-N_ghost,\
                                     :
                                    ]

    # Takes care of boundary conditions and interzonal communications:
    self._da.globalToLocal(self._glob_field, self._local_field)

    # Converting back to af.Array
    self.E1 = af.to_array((local_value[:])[:, :, 0])
    self.E2 = af.to_array((local_value[:])[:, :, 1])
    self.E3 = af.to_array((local_value[:])[:, :, 2])

    self.B1 = af.to_array((local_value[:])[:, :, 3])
    self.B2 = af.to_array((local_value[:])[:, :, 4])
    self.B3 = af.to_array((local_value[:])[:, :, 5])

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

    moment = af.sum(af.sum(af.sum(self.f * moment_variable, 3)*self.dp3, 2)*self.dp2, 1)*self.dp1

    af.eval(moment)
    return(moment)

  # Injection of solver functions into class as methods:
