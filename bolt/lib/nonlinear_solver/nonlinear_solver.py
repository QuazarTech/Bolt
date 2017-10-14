#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the module which contains the functions of the
nonlinear solver of Bolt. It uses a semi-lagrangian 
method based on Cheng-Knorr(1978)
"""

# Importing dependencies:
import arrayfire as af
import numpy as np
import petsc4py, sys

petsc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc
from prettytable import PrettyTable
import socket

# Importing solver libraries:
import bolt.lib.nonlinear_solver.communicate as communicate
import bolt.lib.nonlinear_solver.apply_boundary_conditions as apply_boundary_conditions
import bolt.lib.nonlinear_solver.timestepper as timestepper
import bolt.lib.nonlinear_solver.dump as dump
from bolt.lib.nonlinear_solver.tests.performance.bandwidth_test import bandwidth_test

from bolt.lib.nonlinear_solver.compute_moments \
    import compute_moments as compute_moments_imported
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import fft_poisson, compute_electrostatic_fields

# The following function is used in formatting for print:
def indent(txt, stops=1):
    """
    This function indents every line of the input as a multiple of
    4 spaces.
    """
    return '\n'.join(" " * 4 * stops + line for line in  txt.splitlines())


class nonlinear_solver(object):
    """
    An instance of this class' attributes contains methods which are used
    in evolving the system declared under physical system nonlinearly. The 
    state of the system then may be determined from the attributes of the 
    system such as the distribution function and electromagnetic fields
    """

    def __init__(self, physical_system):
        """
        Constructor for the nonlinear_solver object. It takes the physical
        system object as an argument and uses it in intialization and
        evolution of the system in consideration.
        
        Parameters:
        -----------

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

        # Getting number of ghost zones, and the boundary 
        # conditions that are utilized:
        N_g = self.N_ghost       = physical_system.N_ghost
        self.boundary_conditions = physical_system.boundary_conditions
        
        # Declaring the communicator:
        self._comm = PETSc.COMM_WORLD.tompi4py()

        if(self.physical_system.params.num_devices>1):
            af.set_device(self._comm.rank%self.physical_system.params.num_devices)

        PETSc.Sys.Print('\nBackend Details for Nonlinear Solver:')
        PETSc.Sys.syncPrint(indent('Rank ' + str(self._comm.rank) + ' of ' + str(self._comm.size-1)))
        PETSc.Sys.syncPrint(indent('On Node: '+ socket.gethostname()))
        PETSc.Sys.syncPrint(indent('Device Details:'))
        PETSc.Sys.syncPrint(indent(af.info_str(), 2))
        PETSc.Sys.syncPrint(indent('Device Bandwidth = ' + str(bandwidth_test(100)) + ' GB / sec'))
        PETSc.Sys.syncPrint()
        PETSc.Sys.syncFlush()

        # Defaulting testing flags to false:
        self.testing_source_flag   = False 
        self.performance_test_flag = False

        petsc_bc_in_q1 = 'periodic'
        petsc_bc_in_q2 = 'periodic'

        if(self.boundary_conditions.in_q1 != 'periodic'):
            petsc_bc_in_q1 = 'ghosted'

        if(self.boundary_conditions.in_q2 != 'periodic'):
            petsc_bc_in_q2 = 'ghosted'

        # The DA structure is used in domain decomposition:
        # The following DA is used in the communication routines where
        # information about the data of the distribution function needs
        # to be communicated amongst processes. Additionally this structure
        # automatically takes care of applying periodic boundary conditions.
        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof           = (  self.N_p1 
                                                          * self.N_p2 
                                                          * self.N_p3
                                                         ),
                                         stencil_width = self.N_ghost,
                                         boundary_type = (petsc_bc_in_q1,
                                                          petsc_bc_in_q2
                                                         ),
                                         proc_sizes    = (PETSc.DECIDE, 
                                                          PETSc.DECIDE
                                                         ),
                                         stencil_type  = 1,
                                         comm          = self._comm
                                        )

        # This DA object is used in the communication routines for the
        # EM field quantities. A DOF of 6 is taken so that the communications,
        # and application of B.C's may be carried out in a single call among
        # all the field quantities(E1, E2, E3, B1, B2, B3)
        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof           = 6,
                                              stencil_width = self.N_ghost,
                                              boundary_type = (petsc_bc_in_q1,
                                                               petsc_bc_in_q2
                                                              ),
                                              proc_sizes    = (PETSc.DECIDE,
                                                               PETSc.DECIDE
                                                              ),
                                              stencil_type  = 1,
                                              comm          = self._comm
                                             )

        # Additionally, a DA object also needs to be created for the KSP solver
        # with a DOF of 1:
        self._da_ksp = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                            stencil_width = self.N_ghost,
                                            boundary_type = (petsc_bc_in_q1,
                                                             petsc_bc_in_q2
                                                            ),
                                            proc_sizes    = (PETSc.DECIDE,
                                                             PETSc.DECIDE
                                                            ),
                                            stencil_type  = 1,
                                            comm          = self._comm)

        self._da_dump_moments = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                    dof        = len(self.
                                                                     physical_system.
                                                                     moment_exponents
                                                                    ),
                                                    proc_sizes = (PETSc.DECIDE,
                                                                  PETSc.DECIDE
                                                                 ),
                                                    comm       = self._comm
                                                   )

        # Creation of the local and global vectors from the DA:
        # This is for the distribution function
        self._glob_f  = self._da_f.createGlobalVec()
        self._local_f = self._da_f.createLocalVec()

        # The following global and local vectors are used in
        # the communication routines for EM fields
        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        # The following vector is used to dump the data to file:
        self._glob_moments = self._da_dump_moments.createGlobalVec()

        # Accessing the values of the global and local Vectors:
        self._local_value_f = self._da_f.getVecArray(self._local_f)
        self._glob_value_f  = self._da_f.getVecArray(self._glob_f)

        self._local_value_fields = self._da_fields.getVecArray(self._local_fields)
        self._glob_value_fields  = self._da_fields.getVecArray(self._glob_fields)

        self._glob_moments_value = self._da_dump_moments.\
                                   getVecArray(self._glob_moments)

        # Setting names for the objects which will then be
        # used as the key identifiers for the HDF5 files:
        PETSc.Object.setName(self._glob_f, 'distribution_function')
        PETSc.Object.setName(self._glob_moments, 'moments')

        # Obtaining the array values of the cannonical variables:
        self.q1_center, self.q2_center = self._calculate_q_center()
        self.p1, self.p2, self.p3      = self._calculate_p_center()

        # Assigning the function object to a method of nonlinear solver:
        self._initialize(physical_system.params)

        # Applying dirichlet boundary conditions:        
        if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):

            self.f[:N_g] = self.physical_system.boundary_conditions.\
                           f_left(self.q1_center, self.q2_center,
                                  self.p1, self.p2, self.p3, 
                                  self.physical_system.params
                                 )[:N_g]

            self.f[-N_g:] = self.physical_system.boundary_conditions.\
                            f_right(self.q1_center, self.q2_center,
                                    self.p1, self.p2, self.p3, 
                                    self.physical_system.params
                                   )[-N_g:]

        if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):
            
            self.f[:, :N_g] = self.physical_system.boundary_conditions.\
                              f_bot(self.q1_center, self.q2_center,
                                    self.p1, self.p2, self.p3, 
                                    self.physical_system.params
                                   )[:, :N_g]
            
            self.f[:, -N_g:] = self.physical_system.boundary_conditions.\
                               f_top(self.q1_center, self.q2_center,
                                     self.p1, self.p2, self.p3, 
                                     self.physical_system.params
                                    )[:, -N_g:]

        # Assigning the value to the PETSc Vecs(for dump at t = 0):
        self._local_value_f[:] = np.array(self.f)
        self._glob_value_f[:]  = np.array(self.f)[N_g:-N_g,N_g:-N_g]

        # Assigning the advection terms along q1 and q2
        self._A_q1 = physical_system.A_q(self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[0]
        self._A_q2 = physical_system.A_q(self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[1]

        # Assigning the function objects to methods of the solver:
        self._A_p = physical_system.A_p

        # Source/Sink term:
        self._source = physical_system.source

        # Initializing a variable to track time-elapsed:
        self.time_elapsed = 0

    def _convert_to_q_expanded(self, array):
        """
        Since we are limited to use 4D arrays due to
        the bound from ArrayFire, we define 2 forms
        which can be used such that the computations may
        carried out along all dimensions necessary:

        q_expanded form:(N_q1, N_q2, N_p1 * N_p2 * N_p3)
        p_expanded form:(N_q1 * N_q2, N_p1, N_p2, N_p3)
        
        This function converts the input array from
        p_expanded to q_expanded form.
        """
        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

        array = af.moddims(array,
                           (N_q1_local + 2 * self.N_ghost),
                           (N_q2_local + 2 * self.N_ghost),
                           self.N_p1 * self.N_p2 * self.N_p3
                          )

        af.eval(array)
        return (array)

    def _convert_to_p_expanded(self, array):
        """
        Since we are limited to use 4D arrays due to
        the bound from ArrayFire, we define 2 forms
        which can be used such that the computations may
        carried out along all dimensions necessary:

        q_expanded form:(N_q1, N_q2, N_p1 * N_p2 * N_p3)
        p_expanded form:(N_q1 * N_q2, N_p1, N_p2, N_p3)
        
        This function converts the input array from
        q_expanded to p_expanded form.
        """

        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
        
        array = af.moddims(array,
                             (N_q1_local + 2 * self.N_ghost)
                           * (N_q2_local + 2 * self.N_ghost),
                           self.N_p1, self.N_p2, self.N_p3
                          )

        af.eval(array)
        return (array)

    def _calculate_q_center(self):
        """
        Initializes the cannonical variables q1, q2 using a centered
        formulation. The size, and resolution are the same as declared
        under domain of the physical system object.

        Returns in q_expanded form.
        """

        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

        i_q1_center = i_q1_start + 0.5
        i_q2_center = i_q2_start + 0.5

        i_q1 = (  i_q1_center 
                + np.arange(-self.N_ghost, N_q1_local + self.N_ghost)
               )

        i_q2 = (  i_q2_center
                + np.arange(-self.N_ghost, N_q2_local + self.N_ghost)
               )

        q1_center = self.q1_start + i_q1 * self.dq1
        q2_center = self.q2_start + i_q2 * self.dq2

        q2_center, q1_center = np.meshgrid(q2_center, q1_center)
        q1_center, q2_center = af.to_array(q1_center), af.to_array(q2_center)

        af.eval(q1_center, q2_center)
        return (q1_center, q2_center)

    def _calculate_p_center(self):
        """
        Initializes the cannonical variables p1, p2 and p3 using a centered
        formulation. The size, and resolution are the same as declared
        under domain of the physical system object.

        Returns in q_expanded form.
        """
        p1_center = self.p1_start + (0.5 + np.arange(self.N_p1)) * self.dp1
        p2_center = self.p2_start + (0.5 + np.arange(self.N_p2)) * self.dp2
        p3_center = self.p3_start + (0.5 + np.arange(self.N_p3)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                      p1_center,
                                                      p3_center
                                                     )

        # Flattening the arrays:
        p1_center = af.flat(af.to_array(p1_center))
        p2_center = af.flat(af.to_array(p2_center))
        p3_center = af.flat(af.to_array(p3_center))

        # Reordering such that velocity variation is along
        # axis 2:
        p1_center = af.reorder(p1_center, 2, 3, 0, 1)
        p2_center = af.reorder(p2_center, 2, 3, 0, 1)
        p3_center = af.reorder(p3_center, 2, 3, 0, 1)

        af.eval(p1_center, p2_center, p3_center)
        return (p1_center, p2_center, p3_center)

    def _initialize(self, params):
        """
        Called when the solver object is declared. This function is
        used to initialize the distribution function, and the field
        quantities using the options as provided by the user.
        """
        # Initializing with the provided I.C's:
        # af.broadcast, allows us to perform batched operations 
        # when operating on arrays of different sizes
        # af.broadcast(function, *args) performs batched operations on
        # function(*args)
        self.f = af.broadcast(self.physical_system.initial_conditions.\
                              initialize_f, self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, params
                              )

        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

        # Initializing the EM fields quantities:
        # These quantities are defined for the CK grid:
        # That is at (i + 0.5, j + 0.5):

        # Electric fields are defined at n-th timestep:
        self.E1 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        self.E2 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        self.E3 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        # Magnetic fields are defined at the (n+0.5)-th timestep:
        self.B1 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        self.B2 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        self.B3 = af.constant(0, 
                              N_q1_local + 2 * self.N_ghost,
                              N_q2_local + 2 * self.N_ghost,
                              dtype=af.Dtype.f64
                             )

        # Arrays which hold the magnetic field quantities for the n-th timestep:
        self.B1_n = af.constant(0, 
                                N_q1_local + 2 * self.N_ghost,
                                N_q2_local + 2 * self.N_ghost,
                                dtype=af.Dtype.f64
                               )

        self.B2_n = af.constant(0, 
                                N_q1_local + 2 * self.N_ghost,
                                N_q2_local + 2 * self.N_ghost,
                                dtype=af.Dtype.f64
                               )

        self.B3_n = af.constant(0, 
                                N_q1_local + 2 * self.N_ghost,
                                N_q2_local + 2 * self.N_ghost,
                                dtype=af.Dtype.f64
                               )

        # Declaring the arrays which store data on the FDTD grid:
        self.E1_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        self.E2_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        self.E3_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        self.B1_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        self.B2_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        self.B3_fdtd = af.constant(0, 
                                   N_q1_local + 2 * self.N_ghost,
                                   N_q2_local + 2 * self.N_ghost,
                                   dtype=af.Dtype.f64
                                  )

        if(self.physical_system.params.charge_electron != 0):
            if (self.physical_system.params.fields_initialize == 'fft'):
                fft_poisson(self)

            elif (self.physical_system.params.fields_initialize ==
                  'electrostatic'
                 ):
                compute_electrostatic_fields(self)

            elif (self.physical_system.params.fields_initialize == 'user-defined'):
                self.E1, self.E2, self.E3 = \
                    self.physical_system.initial_conditions.initialize_E(self.q1_center,
                                                                         self.q2_center,
                                                                         params
                                                                        )

                self.B1, self.B2, self.B3 = \
                    self.physical_system.initial_conditions.initialize_B(self.q1_center,
                                                                         self.q2_center,
                                                                         params
                                                                        )

            else:
                raise NotImplementedError('Method not valid/not implemented')

            # Getting the values at the FDTD grid points:
            self.E1_fdtd = 0.5 * (self.E1 + af.shift(self.E1, 0, 1))  # (i+1/2, j)
            self.E2_fdtd = 0.5 * (self.E2 + af.shift(self.E2, 1, 0))  # (i, j+1/2)

            return

    def print_performance_timings(self, N_iters):
        """
        This function is used to check the timings
        of each of the functions which are used during the 
        process of a single-timestep.
        """

        # Initializing the global variables:
        time_ts = np.zeros(1); time_interp2 = np.zeros(1); time_fieldstep = np.zeros(1); 
        time_sourcets = np.zeros(1); time_communicate_f = np.zeros(1); time_fieldsolver = np.zeros(1)
        time_interp3 = np.zeros(1); time_communicate_fields = np.zeros(1) 
        time_apply_bcs_f = np.zeros(1); time_apply_bcs_fields = np.zeros(1)

        # Performing reduction operations to obtain the greatest time amongst nodes/devices:
        self._comm.Reduce(np.array([self.time_ts/N_iters]), time_ts,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_interp2/N_iters]), time_interp2,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_fieldstep/N_iters]), time_fieldstep,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_sourcets/N_iters]), time_sourcets,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_communicate_f/N_iters]), time_communicate_f,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_apply_bcs_f/N_iters]), time_apply_bcs_f,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_fieldsolver/N_iters]), time_fieldsolver,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_interp3/N_iters]), time_interp3,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_communicate_fields/N_iters]), time_communicate_fields,
                          op = MPI.MAX, root = 0
                         )
        self._comm.Reduce(np.array([self.time_apply_bcs_fields/N_iters]), time_apply_bcs_fields,
                          op = MPI.MAX, root = 0
                         )
                         
        if(self._comm.rank == 0):
            table = PrettyTable(["Method", "Time-Taken(s/iter)", "Percentage(%)"])
            table.add_row(['TIMESTEP', time_ts[0]/N_iters, 100])
            
            table.add_row(['Q_ADVECTION', time_interp2[0]/N_iters,
                           100*time_interp2[0]/time_ts[0]
                          ]
                         )
            
            table.add_row(['FIELD-STEP', time_fieldstep[0]/N_iters,
                           100*time_fieldstep[0]/time_ts[0]
                          ]
                         )
            
            table.add_row(['SOURCE_TS', time_sourcets[0]/N_iters,
                           100*time_sourcets[0]/time_ts[0]
                          ]
                         )

            table.add_row(['APPLY_BCS_F', time_apply_bcs_f[0]/N_iters,
                           100*time_apply_bcs_f[0]/time_ts[0]
                          ]
                         )

            table.add_row(['COMMUNICATE_F', time_communicate_f[0]/N_iters,
                           100*time_communicate_f[0]/time_ts[0]
                          ]
                         )
       
            PETSc.Sys.Print(table)

            if(self.physical_system.params.charge_electron != 0):

                PETSc.Sys.Print('FIELDS-STEP consists of:')
                
                table = PrettyTable(["Method", "Time-Taken(s/iter)", "Percentage(%)"])

                table.add_row(['FIELD-STEP', time_fieldstep[0]/N_iters,
                               100
                              ]
                             )

                table.add_row(['FIELD-SOLVER', time_fieldsolver[0]/N_iters,
                               100*time_fieldsolver[0]/time_fieldstep[0]
                              ]
                             )

                table.add_row(['P_ADVECTION', time_interp3[0]/N_iters,
                               100*time_interp3[0]/time_fieldstep[0]
                              ]
                             )

                table.add_row(['APPLY_BCS_FIELDS', time_apply_bcs_fields[0]/N_iters,
                               100*time_apply_bcs_fields[0]/time_fieldstep[0]
                              ]
                             )

                table.add_row(['COMMUNICATE_FIELDS', time_communicate_fields[0]/N_iters,
                               100*time_communicate_fields[0]/time_fieldstep[0]
                              ]
                             )

                PETSc.Sys.Print(table)

            PETSc.Sys.Print('Spatial Zone Cycles/s =', self.N_q1*self.N_q2/time_ts[0])
        
    # Injection of solver functions into class as methods:
    _communicate_f      = communicate.\
                          communicate_f
    _communicate_fields = communicate.\
                          communicate_fields

    _apply_bcs_f      = apply_boundary_conditions.apply_bcs_f
    _apply_bcs_fields = apply_boundary_conditions.apply_bcs_fields

    strang_timestep = timestepper.strang_step
    lie_timestep    = timestepper.lie_step
    swss_timestep   = timestepper.swss_step
    jia_timestep    = timestepper.jia_step

    compute_moments = compute_moments_imported

    dump_distribution_function = dump.dump_distribution_function
    dump_moments               = dump.dump_moments
