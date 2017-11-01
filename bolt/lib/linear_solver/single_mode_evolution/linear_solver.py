#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .dY_dt import dY_dt
from .calculate_dfdp_background import calculate_dfdp_background
from .compute_EM_fields import compute_electrostatic_fields
from .compute_moments import compute_moments as compute_moments_imported
from .timestep import RK4

class linear_solver(object):

    def __init__(self, physical_system):

        self.physical_system = physical_system

        # Storing Domain Information:
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

        # Getting Domain Resolution
        self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1
        self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2
        self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1
        self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2
        self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3

        # Checking that periodic B.C's are utilized:
        if(    physical_system.boundary_conditions.in_q1 != 'periodic' 
            or physical_system.boundary_conditions.in_q2 != 'periodic'
          ):
            raise Exception('Only systems with periodic boundary conditions\
                             can be solved using the linear solver')

        # Intializing position, wavenumber and velocity arrays:
        self.p1, self.p2, self.p3 = self._calculate_p_center()

        # Initializing f, f_hat and the other EM field quantities:
        self._initialize(physical_system.params)

        # This needs to be the linearized collision operator:
        self._source = self.physical_system.source
        print(self.physical_system.source)

    def _calculate_p_center(self):
        """
        Initializes the cannonical variables p1, p2 and p3 using a centered
        formulation. The size, and resolution are the same as declared
        under domain of the physical system object.
        """
        p1_center = \
            self.p1_start + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
        
        p2_center = \
            self.p2_start + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
        
        p3_center = \
            self.p3_start + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                      p1_center,
                                                      p3_center
                                                     )

        p1_center = p1_center.ravel().reshape(1, 1, self.N_p1 * self.N_p2 * self.N_p3)
        p2_center = p2_center.ravel().reshape(1, 1, self.N_p1 * self.N_p2 * self.N_p3)
        p3_center = p3_center.ravel().reshape(1, 1, self.N_p1 * self.N_p2 * self.N_p3)

        return(p1_center, p2_center, p3_center)

    def _initialize(self, params):
        """
        Called when the solver object is declared. This function is
        used to initialize the distribution function, and the field
        quantities using the options as provided by the user. The
        quantities are then mapped to the fourier basis by taking FFTs.
        The independant modes are then evolved by using the linear
        solver.
        """
        # af.broadcast(function, *args) performs batched operations on
        # function(*args):

        q1_center = self.q1_start + (0.5 + np.arange(self.N_q1)) * self.dq1
        q2_center = self.q2_start + (0.5 + np.arange(self.N_q2)) * self.dq2

        q2_center, q1_center = np.meshgrid(q2_center, q1_center)

        q1_center = q1_center.reshape(self.N_q1, self.N_q2, 1)
        q2_center = q2_center.reshape(self.N_q1, self.N_q2, 1)

        f = self.physical_system.initial_conditions.\
            initialize_f(q1_center, q2_center,
                         self.p1, self.p2, self.p3, params
                        )

        self.p1 = self.p1.reshape(self.N_p1, self.N_p2, self.N_p3)
        self.p2 = self.p2.reshape(self.N_p1, self.N_p2, self.N_p3)
        self.p3 = self.p3.reshape(self.N_p1, self.N_p2, self.N_p3)

        # Taking FFT:
        f_hat = np.fft.fft2(f)

        # Since (k_q1, k_q2) = (0, 0) will give the background distribution:
        # The division by (self.N_q1 * self.N_q2) is performed since the FFT
        # at (0, 0) returns (amplitude * (self.N_q1 * self.N_q2))
        # self.f_background = abs(f_hat[0, 0, :])/ (self.N_q1 * self.N_q2)
        # self.f_background = self.f_background.reshape(self.N_p1, self.N_p2, self.N_p3)
        self.f_background =   np.sqrt(1 / (2 * np.pi * 1 * 1)) \
                            * np.exp(-1 * (self.p1)**2 / (2))
        
        calculate_dfdp_background(self)

        self.delta_f_hat =   params.pert_real * self.f_background \
                           + params.pert_imag * self.f_background * 1j 

        self.Y = np.array([self.delta_f_hat])

        compute_electrostatic_fields(self)

        # Using a vector Y to evolve the system:
        self.Y = np.array([self.delta_f_hat, 
                           self.delta_E1_hat, self.delta_E2_hat, self.delta_E3_hat,
                           self.delta_B1_hat, self.delta_B2_hat, self.delta_B3_hat
                          ]
                         )

    _dY_dt          = dY_dt
    compute_moments = compute_moments_imported
    RK4_timestep    = RK4
