#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the module where the main solver object for the
nonlinear solver of bolt is defined. This solver object 
stores the details of the system defined under physical_system, 
and is evolved using the methods of this module.

The solver utilizes a semi-lagrangian which uses advective 
interpolation based on Cheng-Knorr(1978) in the p-space:
   
    - The interpolation schemes available are 
      linear and quadratic spline.

The q-space has the option of using 2 different methods:

- A semi-lagrangian scheme based on Cheng-Knorr(1978) which
  uses advective interpolation.(non-conservative)

    - The interpolation schemes available are 
      linear and quadratic spline.
      TODO:Check with Pavan regarding cubic spline?

- Finite volume scheme(conservative):

    - Riemann solvers available are Lax-Friedrichs and 1st order
      upwind schemes.

    - The reconstruction schemes available are minmod, PPM, and WENO5

"""

# Importing dependencies:
import arrayfire as af
import numpy as np
import petsc4py, sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import socket

# Importing solver libraries:
from . import communicate
from . import apply_boundary_conditions
from . import timestep

from .file_io import dump
from .file_io import load

from .utils.bandwidth_test import bandwidth_test
from .utils.print_with_indent import indent
from .utils.performance_timings import print_table
from .compute_moments import compute_moments as compute_moments_imported
from .EM_fields_solver.electrostatic import fft_poisson, poisson_eqn_3D

class nonlinear_solver(object):
    """
    An instance of this class' attributes contains methods which are used
    in evolving the system declared under physical system nonlinearly. The 
    state of the system then may be determined from the attributes of the 
    system such as the distribution function and electromagnetic fields.
    
    Relevant physical information is obtained by coarse graining this system
    by taking moments of the distribution function. This is performed by the
    compute_moments() method.  
    """

    def __init__(self, physical_system, performance_test_flag = False):
        """
        Constructor for the nonlinear_solver object. It takes the physical
        system object as an argument and uses it in intialization and
        evolution of the system in consideration. 

        Additionally, a performance test flag is also passed which when true 
        stores time which is consumed by each of the major solver routines.
        This proves particularly useful in analyzing performance bottlenecks 
        and obtaining benchmarks.
        
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

        # Printing the backend details for each rank/device/node:
        PETSc.Sys.syncPrint(indent('Rank ' + str(self._comm.rank) + ' of ' + str(self._comm.size-1)))
        PETSc.Sys.syncPrint(indent('On Node: '+ socket.gethostname()))
        PETSc.Sys.syncPrint(indent('Device Details:'))
        PETSc.Sys.syncPrint(indent(af.info_str(), 2))
        PETSc.Sys.syncPrint(indent('Device Bandwidth = ' + str(bandwidth_test(100)) + ' GB / sec'))
        PETSc.Sys.syncPrint()
        PETSc.Sys.syncFlush()

        self.performance_test_flag = performance_test_flag
    
        if(performance_test_flag == True):
        
            self.time_ts                 = 0

            self.time_interp2            = 0
            self.time_sourcets           = 0

            self.time_fvm_solver         = 0
            self.time_reconstruct        = 0
            self.time_riemann            = 0
            
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            
            self.time_apply_bcs_f        = 0
            self.time_apply_bcs_fields   = 0

            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        petsc_bc_in_q1 = 'ghosted'
        petsc_bc_in_q2 = 'ghosted'

        # Only for periodic boundary conditions do the boundary
        # conditions passed to the DA need to be changed. PETSc
        # automatically handles the application of periodic 
        # boundary conditions when running in parallel. In all other
        # cases, ghosted boundaries are used.

        if(self.boundary_conditions.in_q1_left == 'periodic'):
            petsc_bc_in_q1 = 'periodic'

        if(self.boundary_conditions.in_q2_bottom == 'periodic'):
            petsc_bc_in_q2 = 'periodic'

        # DMDA is a data structure to handle a distributed structure 
        # grid and its related core algorithms. It stores metadata of
        # how the grid is partitioned when run in parallel which is 
        # utilized by the various methods of the solver.

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

        # Additionally, a DA object also needs to be created for the SNES solver
        # with a DOF of 1:
	# TODO: Remove the following hardcoded values
        self.length_multiples_q1 = .5
        self.length_multiples_q2 = .25
        self.dq3 = self.dq1
        self.location_in_q3      = 0.3
        self.q3_3D_start =  0.; self.q3_3D_end = 1.3

        self.N_q1_poisson = (2*self.length_multiples_q1+1)*self.N_q1
        self.N_q2_poisson = (2*self.length_multiples_q2+1)*self.N_q2
        self.N_q3_poisson = (int)((self.q3_3D_end - self.q3_3D_start) / self.dq3)
        self.N_ghost_poisson = self.N_ghost

        PETSc.Sys.Print("dq3 = ", self.dq3, "N_q3 = ", self.N_q3_poisson)
        self._da_snes = PETSc.DMDA().create([self.N_q1_poisson, 
	                                     self.N_q2_poisson,
					     self.N_q3_poisson],
                                             stencil_width = self.N_ghost_poisson,
                                             boundary_type = (petsc_bc_in_q1,
                                                              petsc_bc_in_q2,
							      'periodic'
                                                             ),
                                             proc_sizes    = (PETSc.DECIDE,
                                                              PETSc.DECIDE,
                                                              1
                                                             ),
                                             stencil_type  = 0, # Star stencil
                                             dof           = 1,
                                             comm          = self._comm
                                           )
        self.snes = PETSc.SNES().create()
        self.poisson = poisson_eqn_3D(self)
        self.snes.setFunction(self.poisson.compute_residual,
	                      self.poisson.glob_residual
			     )
    
        self.snes.setDM(self._da_snes)
        self.snes.setFromOptions()

        # Obtaining the left-bottom corner coordinates
        # (lowest values of the canonical coordinates in the local zone)
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
        self.i_q1_start = i_q1_start
        self.i_q2_start = i_q2_start
        self.N_q1_local = N_q1_local
        self.N_q2_local = N_q2_local

        # This DA is used by the FileIO routine dump_moments():
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
        # For dumping aux arrays:
        self.dump_aux_arrays_initial_call = 1

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

        # Getting the arrays for the above vectors:
        self._glob_f_array  = self._glob_f.getArray()
        self._local_f_array = self._local_f.getArray()

        self._glob_fields_array  = self._glob_fields.getArray()
        self._local_fields_array = self._local_fields.getArray()

        self._glob_moments_array = self._glob_moments.getArray()

        # Setting names for the objects which will then be
        # used as the key identifiers for the HDF5 files:
        PETSc.Object.setName(self._glob_f, 'distribution_function')
        PETSc.Object.setName(self._glob_moments, 'moments')

        # Obtaining the array values of the cannonical variables:
        self.q1_center, self.q2_center = self._calculate_q_center()
        self.p1, self.p2, self.p3      = self._calculate_p_center()

        # Initialize according to initial condition provided by user:
        self._initialize(physical_system.params)
    
        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
        (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

        # Applying dirichlet boundary conditions:        
        if(self.physical_system.boundary_conditions.in_q1_left == 'dirichlet'):
            # If local zone includes the left physical boundary:
            if(i_q1_start == 0):
                self.f[:, :N_g] = self.boundary_conditions.\
                                  f_left(self.f, self.q1_center, self.q2_center,
                                         self.p1, self.p2, self.p3, 
                                         self.physical_system.params
                                        )[:, :N_g]
    
        if(self.physical_system.boundary_conditions.in_q1_right == 'dirichlet'):
            # If local zone includes the right physical boundary:
            if(i_q1_end == self.N_q1 - 1):
                self.f[:, -N_g:] = self.boundary_conditions.\
                                   f_right(self.f, self.q1_center, self.q2_center,
                                           self.p1, self.p2, self.p3, 
                                           self.physical_system.params
                                          )[:, -N_g:]

        if(self.physical_system.boundary_conditions.in_q2_bottom == 'dirichlet'):
            # If local zone includes the bottom physical boundary:
            if(i_q2_start == 0):
                self.f[:, :, :N_g] = self.boundary_conditions.\
                                     f_bot(self.f, self.q1_center, self.q2_center,
                                           self.p1, self.p2, self.p3, 
                                           self.physical_system.params
                                          )[:, :, :N_g]

        if(self.physical_system.boundary_conditions.in_q2_top == 'dirichlet'):
            # If local zone includes the top physical boundary:
            if(i_q2_end == self.N_q2 - 1):
                self.f[:, :, -N_g:] = self.boundary_conditions.\
                                      f_top(self.f, self.q1_center, self.q2_center,
                                            self.p1, self.p2, self.p3, 
                                            self.physical_system.params
                                           )[:, :, -N_g:]

        # Assigning the value to the PETSc Vecs(for dump at t = 0):
        (af.flat(self.f)).to_ndarray(self._local_f_array)
        (af.flat(self.f[:, N_g:-N_g, N_g:-N_g])).to_ndarray(self._glob_f_array)

        # Assigning the advection terms along q1 and q2
        self._A_q1 = physical_system.A_q(self.q1_center, self.q2_center,
                                         self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[0]
        self._A_q2 = physical_system.A_q(self.q1_center, self.q2_center,
                                         self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[1]

        # Assigning the conservative advection terms along q1 and q2
        self._C_q1 = physical_system.C_q(self.q1_center, self.q2_center,
                                         self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[0]
        self._C_q2 = physical_system.C_q(self.q1_center, self.q2_center,
                                         self.p1, self.p2, self.p3,
                                         physical_system.params
                                        )[1]

        # Assigning the function objects to methods of the solver:
        self._A_p = physical_system.A_p

        # Source/Sink term:
        self._source = physical_system.source

        # Initializing a variable to track time-elapsed:
        # This becomes necessary when applying shearing wall
        # boundary conditions(WIP):
        self.time_elapsed = 0

    def _convert_to_q_expanded(self, array):
        """
        Since we are limited to use 4D arrays due to
        the bound from ArrayFire, we define 2 forms
        which can be used such that the computations may
        carried out along all dimensions necessary:

        q_expanded form:(N_p1 * N_p2 * N_p3, N_q1, N_q2)
        p_expanded form:(N_p1, N_p2, N_p3, N_q1 * N_q2)
        
        This function converts the input array from
        p_expanded to q_expanded form.
        """
        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
     
        array = af.moddims(array,
                           self.N_p1 * self.N_p2 * self.N_p3,
                           (N_q1_local + 2 * self.N_ghost),
                           (N_q2_local + 2 * self.N_ghost)
                          )

        af.eval(array)
        return (array)

    def _convert_to_p_expanded(self, array):
        """
        Since we are limited to use 4D arrays due to
        the bound from ArrayFire, we define 2 forms
        which can be used such that the computations may
        carried out along all dimensions necessary:

        q_expanded form:(N_p1 * N_p2 * N_p3, N_q1, N_q2)
        p_expanded form:(N_p1, N_p2, N_p3, N_q1 * N_q2)
        
        This function converts the input array from
        q_expanded to p_expanded form.
        """
        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
        
        array = af.moddims(array,
                           self.N_p1, self.N_p2, self.N_p3,
                             (N_q1_local + 2 * self.N_ghost)
                           * (N_q2_local + 2 * self.N_ghost)
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

        # Obtaining start coordinates for the local zone
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

        q1_center = af.reorder(q1_center, 2, 0, 1)
        q2_center = af.reorder(q2_center, 2, 0, 1)

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

        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

        # Initializing the EM fields quantities:
        # These quantities are defined for the CK grid:
        # That is at (i + 0.5, j + 0.5):

        # Electric fields are defined at the n-th timestep:
        # Magnetic fields are defined at the (n-1/2)-th timestep:
        self.cell_centered_EM_fields = af.constant(0, 6,
                                                     N_q1_local 
                                                   + 2 * self.N_ghost,
                                                     N_q2_local 
                                                   + 2 * self.N_ghost,
                                                   dtype=af.Dtype.f64
                                                  )

        # Field values at n-th timestep:
        self.cell_centered_EM_fields_at_n = af.constant(0, 6,
                                                          N_q1_local 
                                                        + 2 * self.N_ghost,
                                                          N_q2_local 
                                                        + 2 * self.N_ghost,
                                                        dtype=af.Dtype.f64
                                                       )

        # Field values at (n+1/2)-th timestep:
        self.cell_centered_EM_fields_at_n_plus_half = af.constant(0, 6,
                                                                    N_q1_local 
                                                                  + 2 * self.N_ghost,
                                                                    N_q2_local 
                                                                  + 2 * self.N_ghost,
                                                                  dtype=af.Dtype.f64
                                                                 )


        # Declaring the arrays which store data on the FDTD grid:
        self.yee_grid_EM_fields = af.constant(0, 6,
                                                N_q1_local 
                                              + 2 * self.N_ghost,
                                                N_q2_local 
                                              + 2 * self.N_ghost,
                                              dtype=af.Dtype.f64
                                             )


        if(self.physical_system.params.charge_electron != 0):
            
            if (self.physical_system.params.fields_initialize == 'fft'):
                fft_poisson(self)
                self._communicate_fields()
                self._apply_bcs_fields()

            elif (self.physical_system.params.fields_initialize == 'user-defined'):
                
                E1, E2, E3 = \
                    self.physical_system.initial_conditions.initialize_E(self.q1_center,
                                                                         self.q2_center,
                                                                         params
                                                                        )

                B1, B2, B3 = \
                    self.physical_system.initial_conditions.initialize_B(self.q1_center,
                                                                         self.q2_center,
                                                                         params
                                                                        )

                self.cell_centered_EM_fields  = af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
            
            else:
                raise NotImplementedError('Method not valid/not implemented')

            # Getting the values at the FDTD grid points:
            E1 = self.cell_centered_EM_fields[0] # (i+1/2, j+1/2)
            E2 = self.cell_centered_EM_fields[1] # (i+1/2, j+1/2)
            E3 = self.cell_centered_EM_fields[2] # (i+1/2, j+1/2)
            
            B1 = self.cell_centered_EM_fields[3] # (i+1/2, j+1/2)
            B2 = self.cell_centered_EM_fields[4] # (i+1/2, j+1/2)
            B3 = self.cell_centered_EM_fields[5] # (i+1/2, j+1/2)

            self.yee_grid_EM_fields[0] = 0.5 * (E1 + af.shift(E1, 0, 0, 1))  # (i+1/2, j)
            self.yee_grid_EM_fields[1] = 0.5 * (E2 + af.shift(E2, 0, 1, 0))  # (i, j+1/2)
            self.yee_grid_EM_fields[2] = 0.25 * (  E3 
                                                 + af.shift(E3, 0, 1, 0)
                                                 + af.shift(E3, 0, 0, 1) 
                                                 + af.shift(E3, 0, 1, 1)
                                                )  # (i, j)

            self.yee_grid_EM_fields[3] = 0.5 * (B1 + af.shift(B1, 0, 1, 0)) # (i, j+1/2)
            self.yee_grid_EM_fields[4] = 0.5 * (B2 + af.shift(B2, 0, 0, 1)) # (i+1/2, j)
            self.yee_grid_EM_fields[5] = B3 # (i+1/2, j+1/2)

            # At t = 0, we take the value of B_{0} = B{1/2}:
            self.cell_centered_EM_fields_at_n = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
            self.cell_centered_EM_fields_at_n_plus_half = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
        
    # Injection of solver functions into class as methods:
    _communicate_f      = communicate.\
                          communicate_f

    _communicate_fields = communicate.\
                          communicate_fields

    _apply_bcs_f      = apply_boundary_conditions.apply_bcs_f
    _apply_bcs_fields = apply_boundary_conditions.apply_bcs_fields

    strang_timestep  = timestep.strang_step
    lie_timestep     = timestep.lie_step
    swss_timestep    = timestep.swss_step
    jia_timestep     = timestep.jia_step

    compute_moments = compute_moments_imported

    dump_distribution_function = dump.dump_distribution_function
    dump_moments               = dump.dump_moments
    dump_aux_arrays            = dump.dump_aux_arrays

    load_distribution_function = load.load_distribution_function
    print_performance_timings  = print_table
