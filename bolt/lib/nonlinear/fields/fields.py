#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

from .. import communicate
from .boundaries import apply_bcs_fields

from .electrostatic.fft import fft_poisson
from .electrodynamic.fdtd_explicit import fdtd

class fields_solver(object):
    
    def __init__(physical_system, rho_initial, performance_test_flag = False):
        """
        Constructor for the fields_solver object, which takes in the physical system
        object and initial charge density as the input. 
        
        Additionally, a performance test flag is also passed which when true 
        stores time which is consumed by each of the major solver routines.
        This proves particularly useful in analyzing performance bottlenecks 
        and obtaining benchmarks.

        Parameters:
        -----------
        physical_system: The defined physical system object which holds
                         all the simulation information

        rho_initial: af.Array
                     The initial charge density array that's passed to an electrostatic 
                     solver for initialization

        performance_test_flag: bool
                               When set to true, the time elapsed in routines for the
                               fields solver, inter-processor communication of field values 
                               and application of boundary conditions for the EM_fields
                               is measured.
        """

        self.N_q1 = N_q1
        self.N_q2 = N_q2
        self.N_g  = N_g

        self.q1 = q1
        self.q2 = q2

        self.dq1 = dq1
        self.dq2 = dq2

        self._comm = comm

        self.boundary_conditions = boundary_conditions
        self.params              = params

        self.performance_test_flag   = performance_test_flag
        self.time_fieldsolver        = 0
        self.time_apply_bcs_fields   = 0
        self.time_communicate_fields = 0
        
        self.initialize_E = initialize.initialize_E
        self.initialize_B = initialize.initialize_B
        
        petsc_bc_in_q1 = 'ghosted'
        petsc_bc_in_q2 = 'ghosted'

        # Only for periodic boundary conditions or shearing-box boundary conditions 
        # do the boundary conditions passed to the DA need to be changed. PETSc
        # automatically handles the application of periodic boundary conditions when
        # running in parallel. For shearing box boundary conditions, an interpolation
        # operation needs to be applied on top of the periodic boundary conditions.
        # In all other cases, ghosted boundaries are used.
        
        if(   self.boundary_conditions.in_q1_left == 'periodic'
           or self.boundary_conditions.in_q1_left == 'shearing-box'
          ):
            petsc_bc_in_q1 = 'periodic'

        if(   self.boundary_conditions.in_q2_bottom == 'periodic'
           or self.boundary_conditions.in_q2_bottom == 'shearing-box'
          ):
            petsc_bc_in_q2 = 'periodic'

        nproc_in_q1 = PETSc.DECIDE  
        nproc_in_q2 = PETSc.DECIDE

        # Since shearing boundary conditions require interpolations which are non-local:
        if(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            nproc_in_q1 = 1
        
        if(self.boundary_conditions.in_q1_left == 'shearing-box'):
            nproc_in_q2 = 1

        # This DA object is used in the communication routines for the
        # EM field quantities. A DOF of 6 is taken so that the communications,
        # and application of B.C's may be carried out in a single call among
        # all the field quantities(E1, E2, E3, B1, B2, B3)
        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof           = 6,
                                              stencil_width = self.N_g,
                                              boundary_type = (petsc_bc_in_q1,
                                                               petsc_bc_in_q2
                                                              ),
                                              proc_sizes    = (nproc_in_q1, 
                                                               nproc_in_q2
                                                              ),
                                              stencil_type  = 1,
                                              comm          = self._comm
                                             )

        # The following global and local vectors are used in
        # the communication routines for EM fields
        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        self._glob_fields_array  = self._glob_fields.getArray()
        self._local_fields_array = self._local_fields.getArray()

        PETSc.Object.setName(self._glob_fields, 'EM_fields')
        
        # Alternating upon each call to get_fields for FVM:
        # This ensures that the fields are staggerred correctly in time:
        self.at_n = True
        
        self._initialize(rho_initial)
    
    def _initialize(self, rho_initial):
        """
        Called when the solver object is declared. This function is
        used to initialize the EM field quantities using the options 
        as provided by the user.

        Parameters
        ----------

        rho_initial : af.Array
                     The initial charge density array that's passed to an electrostatic 
                     solver for initialization
        """
        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

        # Following quantities are cell-centered (i + 0.5, j + 0.5):
        # Electric fields are defined at the n-th timestep:
        # Magnetic fields are defined at the (n-1/2)-th timestep:
        self.cell_centered_EM_fields = af.constant(0, 6, 1, 
                                                   N_q1_local + 2 * self.N_g,
                                                   N_q2_local + 2 * self.N_g,
                                                   dtype=af.Dtype.f64
                                                  )

        # Field values at n-th timestep:
        self.cell_centered_EM_fields_at_n = af.constant(0, 6, 1, 
                                                        N_q1_local + 2 * self.N_g,
                                                        N_q2_local + 2 * self.N_g,
                                                        dtype=af.Dtype.f64
                                                       )


        # Field values at (n+1/2)-th timestep:
        self.cell_centered_EM_fields_at_n_plus_half = af.constant(0, 6, 1, 
                                                                  N_q1_local + 2 * self.N_g,
                                                                  N_q2_local + 2 * self.N_g,
                                                                  dtype=af.Dtype.f64
                                                                 )

        # Declaring the arrays which store data on the yee grid for FDTD:
        self.yee_grid_EM_fields = af.constant(0, 6, 1, 
                                              N_q1_local + 2 * self.N_g,
                                              N_q2_local + 2 * self.N_g,
                                              dtype=af.Dtype.f64
                                             )

        if(self.params.fields_type == 'user-defined'):
            try:
                assert(self.params.fields_initialize == 'user-defined')
            except:
                raise Exception('It is expected that the fields initialization method is also \
                                 userdefined when the fields type is declared to be userdefined'
                               )
        
        if (self.params.fields_initialize == 'fft'):
            fft_poisson(self, rho_initial)
            communicate.communicate_fields(self)
            apply_bcs_fields(self)

        elif (self.nls.physical_system.params.fields_initialize == 'user-defined'):

            if(self.nls.physical_system.params.fields_type != 'user-defined'):            
                E1, E2, E3 = self.initialize_E(self.q1,
                                               self.q2,
                                               self.params
                                              )

                B1, B2, B3 = self.initialize_B(self.q1,
                                               self.q2,
                                               self.params
                                              )

            else:

                E1, E2, E3 = self.params.user_defined_E(self.q1,
                                                        self.q2,
                                                        0
                                                       )

                B1, B2, B3 = self.params.user_defined_B(self.q1,
                                                        self.q2,
                                                        0
                                                       )

            self.cell_centered_EM_fields = af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
        
        else:
            raise NotImplementedError('Method not valid/not implemented')

        self.cell_centered_grid_to_yee_grid()
        
        # At t = 0, we take the value of B_{0} = B{1/2}:
        self.cell_centered_EM_fields_at_n = self.cell_centered_EM_fields
        self.cell_centered_EM_fields_at_n_plus_half = self.cell_centered_EM_fields

    def cell_centered_grid_to_yee_grid(self):

        E1 = self.cell_centered_EM_fields[0]
        E2 = self.cell_centered_EM_fields[1]
        E3 = self.cell_centered_EM_fields[2]

        B1 = self.cell_centered_EM_fields[3]
        B2 = self.cell_centered_EM_fields[4]
        B3 = self.cell_centered_EM_fields[5]

        self.yee_grid_EM_fields[0] = 0.5 * (E1 + af.shift(E1, 0, 0, 0, 1))  # (i+1/2, j)
        self.yee_grid_EM_fields[1] = 0.5 * (E2 + af.shift(E2, 0, 0, 1, 0))  # (i, j+1/2)
        self.yee_grid_EM_fields[2] = 0.25 * (  E3 
                                             + af.shift(E3, 0, 0, 1, 0)
                                             + af.shift(E3, 0, 0, 0, 1) 
                                             + af.shift(E3, 0, 0, 1, 1)
                                            )  # (i, j)

        self.yee_grid_EM_fields[3] = 0.5 * (B1 + af.shift(B1, 0, 0, 1, 0)) # (i, j+1/2)
        self.yee_grid_EM_fields[4] = 0.5 * (B2 + af.shift(B2, 0, 0, 0, 1)) # (i+1/2, j)
        self.yee_grid_EM_fields[5] = B3 # (i+1/2, j+1/2)

        af.eval(self.yee_grid_EM_fields)
        return

    def yee_grid_to_cell_centered_grid(self):
        
        E1_yee = self.yee_grid_EM_fields[0] # (i + 1/2, j)
        E2_yee = self.yee_grid_EM_fields[1] # (i, j + 1/2)
        E3_yee = self.yee_grid_EM_fields[2] # (i, j)

        B1_yee = self.yee_grid_EM_fields[3] # (i, j + 1/2)
        B2_yee = self.yee_grid_EM_fields[4] # (i + 1/2, j)
        B3_yee = self.yee_grid_EM_fields[5] # (i + 1/2, j + 1/2)

        # Interpolating at the (i + 1/2, j + 1/2) point of the grid:
        self.cell_centered_EM_fields[0] = 0.5 * (E1_yee + af.shift(E1_yee, 0, 0,  0, -1))
        self.cell_centered_EM_fields[1] = 0.5 * (E2_yee + af.shift(E2_yee, 0, 0, -1,  0))
        self.cell_centered_EM_fields[2] = 0.25 * (  E3_yee 
                                                  + af.shift(E3_yee, 0, 0,  0, -1)
                                                  + af.shift(E3_yee, 0, 0, -1,  0)
                                                  + af.shift(E3_yee, 0, 0, -1, -1)
                                                 )

        self.cell_centered_EM_fields[3] = 0.5 * (B1_yee + af.shift(B1_yee, 0, 0, -1,  0))
        self.cell_centered_EM_fields[4] = 0.5 * (B2_yee + af.shift(B2_yee, 0, 0,  0, -1))
        self.cell_centered_EM_fields[5] = B3_yee

        af.eval(self.cell_centered_EM_fields)
        return

    def current_values_to_yee_grid(self):

        # Obtaining the values for current density on the Yee-Grid:
        self.J1 = 0.5 * (self.J1 + af.shift(self.J1, 0, 0, 0, 1))  # (i + 1/2, j)
        self.J2 = 0.5 * (self.J2 + af.shift(self.J2, 0, 0, 1, 0))  # (i, j + 1/2)

        self.J3 = 0.25 * (  self.J3 + af.shift(self.J3, 0, 0, 1, 0)
                          + af.shift(self.J3, 0, 0, 0, 1)
                          + af.shift(self.J3, 0, 0, 1, 1)
                         )  # (i, j)

        return

    def compute_electrostatic_fields(self, rho):

        if (self.params.fields_initialize == 'fft'):
            
            fft_poisson(self, rho)
            communicate.communicate_fields(self)
            apply_bcs_fields(self)

        # ADD SNES BELOW

    def evolve_electrodynamic_fields(self, J1, J2, J3, dt):
        """
        Evolve the fields using FDTD.

        Parameters
        ----------

        J1 : af.Array
             Array which contains the J1 current for each species.        
        
        J2 : af.Array
             Array which contains the J2 current for each species.        
        
        J3 : af.Array
             Array which contains the J3 current for each species.        
        
        dt: double
            Timestep size
        """

        self.J1 = af.sum(J1, 1)
        self.J2 = af.sum(J2, 1)
        self.J3 = af.sum(J3, 1)

        self.current_values_to_yee_grid()

        # Here:
        # cell_centered_EM_fields[:3] is at n
        # cell_centered_EM_fields[3:] is at n+1/2
        # cell_centered_EM_fields_at_n_plus_half[3:] is at n-1/2

        self.cell_centered_EM_fields_at_n[:3] = self.cell_centered_EM_fields[:3]
        self.cell_centered_EM_fields_at_n[3:] = \
            0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[3:] 
                   + self.cell_centered_EM_fields[3:]
                  )

        self.cell_centered_EM_fields_at_n_plus_half[3:] = self.cell_centered_EM_fields[3:]

        fdtd(self, dt)
        self.yee_grid_to_cell_centered_grid()

        # Here
        # cell_centered_EM_fields[:3] is at n+1
        # cell_centered_EM_fields[3:] is at n+3/2

        self.cell_centered_EM_fields_at_n_plus_half[:3] = \
            0.5 * (  self.cell_centered_EM_fields_at_n[:3] 
                   + self.cell_centered_EM_fields[:3]
                  )

        return

    def update_user_defined_fields(self, time_elapsed):
        """
        Updates the cell-centered EM fields value using the value that is 
        returned by the user defined function at that particular time.

        Parameters
        ----------

        time_elapsed : double
                       Time at which the field values are to be evaluated.
        """

        E1, E2, E3 = self.params.user_defined_E(self.q1,
                                                self.q2,
                                                time_elapsed
                                               )

        B1, B2, B3 = self.params.user_defined_B(self.q1,
                                                self.q2,
                                                time_elapsed
                                               )

        self.cell_centered_EM_fields = af.join(0, E1, E2, E3, 
                                               af.join(0, B1, B2, B3)
                                              )

        return

    def get_fields(self):
        """
        Returns the fields value as held by the
        solver in it's current state.
        """
        if(self.params.fields_solver != 'fdtd'):

            E1 = self.cell_centered_EM_fields[0]
            E2 = self.cell_centered_EM_fields[1]
            E3 = self.cell_centered_EM_fields[2]

            B1 = self.cell_centered_EM_fields[3]
            B2 = self.cell_centered_EM_fields[4]
            B3 = self.cell_centered_EM_fields[5]

        else:
            if(self.at_n == True):

                E1 = self.cell_centered_EM_fields_at_n[0]
                E2 = self.cell_centered_EM_fields_at_n[1]
                E3 = self.cell_centered_EM_fields_at_n[2]

                B1 = self.cell_centered_EM_fields_at_n[3]
                B2 = self.cell_centered_EM_fields_at_n[4]
                B3 = self.cell_centered_EM_fields_at_n[5]

            else:

                E1 = self.cell_centered_EM_fields_at_n_plus_half[0]
                E2 = self.cell_centered_EM_fields_at_n_plus_half[1]
                E3 = self.cell_centered_EM_fields_at_n_plus_half[2]

                B1 = self.cell_centered_EM_fields_at_n_plus_half[3]
                B2 = self.cell_centered_EM_fields_at_n_plus_half[4]
                B3 = self.cell_centered_EM_fields_at_n_plus_half[5]

        if(self.params.solver_method_in_p == 'FVM'):
            # Alternating upon each call for FVM:
            # TEMP FIX: Need to change to something more clean
            self.at_n = not(self.at_n)

        return(E1, E2, E3, B1, B2, B3)
