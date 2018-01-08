#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

from .. import communicate
from .. import apply_boundary_conditions

from .electrostatic_solvers.fft import fft_poisson
from .electrodynamic_solvers.fdtd_explicit import fdtd, yee_grid_to_cell_centered_grid

class fields_solver(object):
    
    def __init__(self, nls_object):

        self.nls = nls_object

        petsc_bc_in_q1 = 'ghosted'
        petsc_bc_in_q2 = 'ghosted'

        # Only for periodic boundary conditions or shearing-box boundary conditions 
        # do the boundary conditions passed to the DA need to be changed. PETSc
        # automatically handles the application of periodic boundary conditions when
        # running in parallel. For shearing box boundary conditions, an interpolation
        # operation needs to be applied on top of the periodic boundary conditions.
        # In all other cases, ghosted boundaries are used.
        
        if(   self.nls.boundary_conditions.in_q1_left == 'periodic'
           or self.nls.boundary_conditions.in_q1_left == 'shearing-box'
          ):
            petsc_bc_in_q1 = 'periodic'

        if(   self.nls.boundary_conditions.in_q2_bottom == 'periodic'
           or self.nls.boundary_conditions.in_q2_bottom == 'shearing-box'
          ):
            petsc_bc_in_q2 = 'periodic'

        nproc_in_q1 = PETSc.DECIDE  
        nproc_in_q2 = PETSc.DECIDE

        # Since shearing boundary conditions require interpolations which are non-local:
        if(self.nls.boundary_conditions.in_q2_bottom == 'shearing-box'):
            nproc_in_q1 = 1
        
        if(self.nls.boundary_conditions.in_q1_left == 'shearing-box'):
            nproc_in_q2 = 1

        # This DA object is used in the communication routines for the
        # EM field quantities. A DOF of 6 is taken so that the communications,
        # and application of B.C's may be carried out in a single call among
        # all the field quantities(E1, E2, E3, B1, B2, B3)
        self._da_fields = PETSc.DMDA().create([self.nls.N_q1, self.nls.N_q2],
                                              dof           = 6,
                                              stencil_width = self.nls.N_ghost_q,
                                              boundary_type = (petsc_bc_in_q1,
                                                               petsc_bc_in_q2
                                                              ),
                                              proc_sizes    = (nproc_in_q1, 
                                                               nproc_in_q2
                                                              ),
                                              stencil_type  = 1,
                                              comm          = self.nls._comm
                                             )

        # The following global and local vectors are used in
        # the communication routines for EM fields
        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        self._glob_fields_array  = self._glob_fields.getArray()
        self._local_fields_array = self._local_fields.getArray()

        PETSc.Object.setName(self._glob_fields, 'EM_fields')
        self.at_n = True
        self._initialize()
    
    def _initialize(self):
        """
        Called when the solver object is declared. This function is
        used to initialize the field quantities
        """

        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

        # Following quantities are cell-centered (i + 0.5, j + 0.5):
        # Electric fields are defined at the n-th timestep:
        # Magnetic fields are defined at the (n-1/2)-th timestep:
        self.cell_centered_EM_fields = af.constant(0, 6, 1, 
                                                     N_q1_local 
                                                   + 2 * self.nls.N_ghost_q,
                                                     N_q2_local 
                                                   + 2 * self.nls.N_ghost_q,
                                                   dtype=af.Dtype.f64
                                                  )

        # Field values at n-th timestep:
        self.cell_centered_EM_fields_at_n = af.constant(0, 6, 1, 
                                                          N_q1_local 
                                                        + 2 * self.nls.N_ghost_q,
                                                          N_q2_local 
                                                        + 2 * self.nls.N_ghost_q,
                                                        dtype=af.Dtype.f64
                                                       )

        # Field values at (n+1/2)-th timestep:
        self.cell_centered_EM_fields_at_n_plus_half = af.constant(0, 6, 1,
                                                                    N_q1_local 
                                                                  + 2 * self.nls.N_ghost_q,
                                                                    N_q2_local 
                                                                  + 2 * self.nls.N_ghost_q,
                                                                  dtype=af.Dtype.f64
                                                                 )


        # Declaring the arrays which store data on the yee grid for FDTD:
        self.yee_grid_EM_fields = af.constant(0, 6, 1,
                                                N_q1_local 
                                              + 2 * self.nls.N_ghost_q,
                                                N_q2_local 
                                              + 2 * self.nls.N_ghost_q,
                                              dtype=af.Dtype.f64
                                             )

        if(self.nls.physical_system.params.fields_type == 'user-defined'):
            try:
                assert(self.nls.physical_system.params.fields_initialize == 'user-defined')
            except:
                raise Exception('It is expected that the fields initialization method is also \
                                 userdefined when the fields type is declared to be userdefined'
                               )
        
        if (self.nls.physical_system.params.fields_initialize == 'fft'):
            fft_poisson(self)
            communicate.communicate_fields(self)
            apply_boundary_conditions.apply_bcs_fields(self)

        elif (self.nls.physical_system.params.fields_initialize == 'user-defined'):

            if(self.nls.physical_system.params.fields_type != 'user-defined'):            
                E1, E2, E3 = \
                    self.physical_system.initial_conditions.initialize_E(self.nls.q1_center,
                                                                         self.nls.q2_center,
                                                                         self.nls.physical_system.params
                                                                        )

                B1, B2, B3 = \
                    self.physical_system.initial_conditions.initialize_B(self.q1_center,
                                                                         self.q2_center,
                                                                         self.nls.physical_system.params
                                                                        )
            else:

                E1, E2, E3 = \
                    self.physical_system.params.user_defined_E(self.nls.q1_center,
                                                               self.nls.q2_center,
                                                               0
                                                              )

                B1, B2, B3 = \
                    self.physical_system.params.user_defined_B(self.nls.q1_center,
                                                               self.nls.q2_center,
                                                               0
                                                              )

            self.cell_centered_EM_fields = af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
        
        else:
            raise NotImplementedError('Method not valid/not implemented')

        E1 = self.cell_centered_EM_fields[0]
        E2 = self.cell_centered_EM_fields[1]
        E3 = self.cell_centered_EM_fields[2]

        B1 = self.cell_centered_EM_fields[3]
        B2 = self.cell_centered_EM_fields[4]
        B3 = self.cell_centered_EM_fields[5]

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


    def evolve_fields(self, dt):
        
        if(self.nls.physical_system.params.fields_solver == 'fdtd'):
            self.J1 =   af.sum(self.nls.physical_system.params.charge) \
                      * self.nls.compute_moments('mom_v1_bulk')  # (i + 1/2, j + 1/2)
            self.J2 =   af.sum(self.nls.physical_system.params.charge) \
                      * self.nls.compute_moments('mom_v2_bulk')  # (i + 1/2, j + 1/2)
            self.J3 =   af.sum(self.nls.physical_system.params.charge) \
                      * self.nls.compute_moments('mom_v3_bulk')  # (i + 1/2, j + 1/2)

            self.J1 = af.sum(self.J1, 1)
            self.J2 = af.sum(self.J2, 1)
            self.J3 = af.sum(self.J3, 1)

            # Obtaining the values for current density on the Yee-Grid:
            self.J1 = 0.5 * (self.J1 + af.shift(self.J1, 0, 0, 0, 1))  # (i + 1/2, j)
            self.J2 = 0.5 * (self.J2 + af.shift(self.J2, 0, 0, 1, 0))  # (i, j + 1/2)

            self.J3 = 0.25 * (  self.J3 + af.shift(self.J3, 0, 0, 1, 0)
                              + af.shift(self.J3, 0, 0, 0, 1)
                              + af.shift(self.J3, 0, 0, 1, 1)
                             )  # (i, j)

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
            yee_grid_to_cell_centered_grid(self)

            # Here
            # cell_centered_EM_fields[:3] is at n+1
            # cell_centered_EM_fields[3:] is at n+3/2

            self.cell_centered_EM_fields_at_n_plus_half[:3] = \
                0.5 * (  self.cell_centered_EM_fields_at_n[:3] 
                       + self.cell_centered_EM_fields[:3]
                      )

        elif(self.nls.physical_system.params.fields_type == 'user-defined'
             and self.at_n == True
            ):
            
            E1, E2, E3 = \
                self.physical_system.params.user_defined_E(self.q1_center,
                                                           self.q2_center,
                                                           self.time_elapsed
                                                          )

            B1, B2, B3 = \
                self.physical_system.params.user_defined_B(self.q1_center,
                                                           self.q2_center,
                                                           self.time_elapsed
                                                          )

            self.cell_centered_EM_fields  = af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

        elif(self.nls.physical_system.params.fields_type == 'user-defined'
             and self.at_n == False
            ):
            
            E1, E2, E3 = \
                self.physical_system.params.user_defined_E(self.q1_center,
                                                           self.q2_center,
                                                           self.time_elapsed + 0.5 * self.dt
                                                          )

            B1, B2, B3 = \
                self.physical_system.params.user_defined_B(self.q1_center,
                                                           self.q2_center,
                                                           self.time_elapsed + 0.5 * self.dt
                                                          )

            self.cell_centered_EM_fields  = af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

        else:
            if(self.nls.physical_system.params.fields_solver == 'fft'):
                fft_poisson(self)
                communicate.communicate_fields(self)
                apply_boundary_conditions.apply_bcs_fields(self)

            else:
                raise NotImplementedError('The method specified is \
                                           invalid/not-implemented'
                                         )

    def get_fields(self):

        if(self.nls.physical_system.params.fields_solver == 'fft'):

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

        # Alternating upon each call
        # TEMP FIX: Need to change to something more clear
        self.at_n = not(self.at_n)

        return(E1, E2, E3, B1, B2, B3)
