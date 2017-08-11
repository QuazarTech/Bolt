#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

# Importing solver libraries:
from lib.nonlinear_solver.communicate \
    import communicate_distribution_function, communicate_fields

from lib.nonlinear_solver.timestepper import strang_step, lie_step
from lib.nonlinear_solver.compute_moments \
    import compute_moments as compute_moments_imported
from lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import fft_poisson, compute_electrostatic_fields


class nonlinear_solver(object):
    """
    An instance of this class is used in evolving system as
    defined under the physical_system object. The methods of this
    object evolve the system using a semi-lagrangian method based
    on Cheng-Knorr(1978)
    """

    def __init__(self, physical_system):
        """
        physical_system: The defined physical system object which holds
                         all the simulation information such as the initial
                         conditions, and the domain info is passed as an
                         argument in defining an instance of the
                         nonlinear_solver. This system is then evolved, and
                         monitored using the various methods under the
                         nonlinear_solver class.
        """
        self.physical_system = physical_system

        # Holding Domain Info:
        self.q1_start, self.q1_end = physical_system.q1_start,\
                                     physical_system.q1_end
        self.q2_start, self.q2_end = physical_system.q2_start,\
                                     physical_system.q2_end
        self.p1_start, self.p1_end = physical_system.p1_start,\
                                     physical_system.p1_end
        self.p2_start, self.p2_end = physical_system.p2_start,\
                                     physical_system.p2_end
        self.p3_start, self.p3_end = physical_system.p3_start,\
                                     physical_system.p3_end

        # Holding Domain Resolution:
        self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1
        self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2
        self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1
        self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2
        self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3

        # Getting number of ghost zones, and the boundary conditions that are
        # utilized
        self.N_ghost = physical_system.N_ghost
        self.bc_in_q1, self.bc_in_q2 = physical_system.bc_in_q1,\
                                       physical_system.bc_in_q2

        # Declaring the communicator:
        self._comm = PETSc.COMM_WORLD.tompi4py()

        # The DA structure is used in domain decomposition:
        # The following DA is used in the communication routines where
        # information about the data of the distribution function needs
        # to be communicated amongst processes. Additionally this structure
        # automatically takes care of applying periodic boundary conditions.

        self._da = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                       dof=(self.N_p1 * self.N_p2 * self.N_p3),
                                       stencil_width=self.N_ghost,
                                       boundary_type=(self.bc_in_q1,
                                                      self.bc_in_q2),
                                       proc_sizes=(PETSc.DECIDE, PETSc.DECIDE),
                                       stencil_type=1,
                                       comm=self._comm)

        # This DA object is used in the communication routines for the
        # EM field quantities. A DOF of 6 is taken so that the communications,
        # and application of B.C's may be carried out in a single call among
        # all the field quantities(E1, E2, E3, B1, B2, B3)
        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof=6,
                                              stencil_width=self.N_ghost,
                                              boundary_type=(self.bc_in_q1,
                                                             self.bc_in_q2),
                                              proc_sizes=(PETSc.DECIDE,
                                                          PETSc.DECIDE),
                                              stencil_type=1,
                                              comm=self._comm)

        # Additionally, a DA object also needs to be created for the KSP solver
        # with a DOF of 1:
        self._da_ksp = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                            stencil_width=self.N_ghost,
                                            boundary_type=(self.bc_in_q1,
                                                           self.bc_in_q2),
                                            proc_sizes=(PETSc.DECIDE,
                                                        PETSc.DECIDE),
                                            stencil_type=1,
                                            comm=self._comm)

        # Creation of the local and global vectors from the DA:
        # This is for the distribution function
        self._glob = self._da.createGlobalVec()
        self._local = self._da.createLocalVec()

        # The following global and local vectors are used in
        # the communication routines for EM fields
        self._glob_fields = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        # Obtaining the array values of the cannonical variables:
        self.q1_center, self.q2_center = self._calculate_q_center()

        self.p1, self.p2, self.p3 = self._calculate_p_center()

        # Assigning the function object to a method of nonlinear solver:
        self._initialize(physical_system.params)

        # Assigning the advection terms along q1 and q2
        self._A_q1 = physical_system.A_q(self.p1, self.p2, self.p3,
                                         physical_system.params)[0]
        self._A_q2 = physical_system.A_q(self.p1, self.p2, self.p3,
                                         physical_system.params)[1]

        # Assigning the function objects to methods of the solver:
        self._A_p = physical_system.A_p

        # Source/Sink term(Restrict to relaxation type collision operators):
        self._source_or_sink = physical_system.source_or_sink

    def _convert_to_qExpand(self, array):
        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_lowest, i_q2_lowest),
         (N_q1_local, N_q2_local)) = self._da.getCorners()
        array = af.moddims(array,
                           (N_q1_local + 2 * self.N_ghost),
                           (N_q2_local + 2 * self.N_ghost),
                           self.N_p1 * self.N_p2 * self.N_p3, 1)

        af.eval(array)
        return (array)

    def _convert_to_pExpand(self, array):
        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_lowest, i_q2_lowest),
         (N_q1_local, N_q2_local)) = self._da.getCorners()
        array = af.moddims(array,
                           (N_q1_local + 2 * self.N_ghost) *
                           (N_q2_local + 2 * self.N_ghost),
                           self.N_p1, self.N_p2, self.N_p3)

        af.eval(array)
        return (array)

    def _calculate_q_center(self):
        """
        Used in initializing the cannonical variables q1, q2
        """
        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_lowest, i_q2_lowest),
         (N_q1_local, N_q2_local)) = self._da.getCorners()

        i_q1_center = i_q1_lowest + 0.5
        i_q2_center = i_q2_lowest + 0.5

        i_q1 = i_q1_center + \
               np.arange(-self.N_ghost, N_q1_local + self.N_ghost)
        i_q2 = i_q2_center + \
               np.arange(-self.N_ghost, N_q2_local + self.N_ghost)

        q1_center = af.to_array(self.q1_start + i_q1 * self.dq1)
        q2_center = af.to_array(self.q2_start + i_q2 * self.dq2)

        # Tiling such that variation in q1 is along axis 0:
        q1_center = af.tile(q1_center, 1, N_q2_local + 2 * self.N_ghost,
                            self.N_p1 * self.N_p2 * self.N_p3)

        # Tiling such that variation in q2 is along axis 1:
        q2_center = af.tile(af.reorder(q2_center),
                            N_q1_local + 2 * self.N_ghost, 1,
                            self.N_p1 * self.N_p2 * self.N_p3, 1)

        af.eval(q1_center, q2_center)
        # Returns in positionsExpanded form(Nq1, Nq2, Np1*Np2*Np3, 1)
        return (q1_center, q2_center)

    def _calculate_p_center(self):
        """
        Used in initializing the cannonical variables p1, p2, p3
        """
        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_lowest, i_q2_lowest),
         (N_q1_local, N_q2_local)) = self._da.getCorners()

        N_ghost = self.N_ghost

        p1_center = self.p1_start + \
                    (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
        p2_center = self.p2_start + \
                    (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
        p3_center = self.p3_start + \
                    (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                      p1_center,
                                                      p3_center)

        p1_center = af.flat(af.to_array(p1_center))
        p2_center = af.flat(af.to_array(p2_center))
        p3_center = af.flat(af.to_array(p3_center))

        p1_center = af.tile(af.reorder(p1_center, 2, 3, 0, 1),
                            N_q1_local + 2 * N_ghost,
                            N_q2_local + 2 * N_ghost, 1, 1)

        p2_center = af.tile(af.reorder(p2_center, 2, 3, 0, 1),
                            N_q1_local + 2 * N_ghost,
                            N_q2_local + 2 * N_ghost, 1, 1)

        p3_center = af.tile(af.reorder(p3_center, 2, 3, 0, 1),
                            N_q1_local + 2 * N_ghost,
                            N_q2_local + 2 * N_ghost, 1, 1)

        af.eval(p1_center, p2_center, p3_center)
        # returned in positionsExpanded form
        # (N_q1, N_q2, N_p1*N_p2*N_p3)
        return (p1_center, p2_center, p3_center)

    # Injection of solver functions into class as methods:
    _communicate_distribution_function = communicate_distribution_function
    _communicate_fields = communicate_fields

    strang_timestep = strang_step
    lie_timestep = lie_step

    compute_moments = compute_moments_imported

    def _initialize(self, params):
        """
        Used to initialize the distribution function, and the
        EM field quantities
        """
        self.f = self.physical_system.initial_conditions.\
            initialize_f(self.q1_center, self.q2_center,
                         self.p1, self.p2, self.p3, params
                         )

        self.E1 = af.constant(0, self.f.shape[0],
                              self.f.shape[1], dtype=af.Dtype.f64)
        self.E2 = self.E1.copy()
        self.E3 = self.E1.copy()

        self.B1 = self.E1.copy()
        self.B2 = self.E1.copy()
        self.B3 = self.E1.copy()

        self.E1_fdtd = self.E1.copy()
        self.E2_fdtd = self.E1.copy()
        self.E3_fdtd = self.E1.copy()

        self.B1_fdtd = self.E1.copy()
        self.B2_fdtd = self.E1.copy()
        self.B3_fdtd = self.E1.copy()

        if (self.physical_system.params.fields_initialize == 'fft'):
            fft_poisson(self)

        elif (self.physical_system.params.fields_initialize ==
              'electrostatic'):
            compute_electrostatic_fields(self)

        elif (self.physical_system.params.fields_initialize == 'user-defined'):
            self.E1, self.E2, self.E3 = self.physical_system.\
                initial_conditions.initialize_E(self.q1_center, self.q2_center,
                                                params)

            self.B1, self.B2, self.B3 = self.physical_system.\
                initial_conditions.initialize_B(self.q1_center, self.q2_center,
                                                params)

        else:
            raise NotImplementedError('Method not valid/not implemented')

        # Getting the values at the FDTD grid points:
        self.E1_fdtd = 0.5 * \
                       (self.E1 + af.shift(self.E1, 0, 1))  # (i + 1/2, j)
        self.E2_fdtd = 0.5 * \
                       (self.E2 + af.shift(self.E2, 1, 0))  # (i, j + 1/2)

        return
