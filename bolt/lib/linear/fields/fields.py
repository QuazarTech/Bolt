#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from bolt.lib.utils.fft_funcs import fft2, ifft2
from bolt.lib.utils.calculate_k import calculate_k
from bolt.lib.utils.calculate_q import calculate_q_center
from .electrostatic_solver import compute_electrostatic_fields

class fields_solver(object):
    
    def __init__(self, physical_system, rho_hat_initial):
        """
        Constructor for the fields_solver object, which takes in the physical system
        object and the FT of the initial charge density as the input. 

        Parameters:
        -----------
        physical_system: The defined physical system object which holds
                         all the simulation information

        rho_hat_initial: af.Array
                         The FT of the initial charge density array that's passed to 
                         an electrostatic solver for initialization
        """

        self.N_q1   = physical_system.N_q1
        self.N_q2   = physical_system.N_q2
        self.dq1    = physical_system.dq1
        self.dq2    = physical_system.dq2
        self.params = physical_system.params

        self.initialize = physical_system.initial_conditions

        self.q1_center, self.q2_center = \
            calculate_q_center(physical_system.q1_start, 
                               physical_system.q2_start,
                               self.N_q1, self.N_q2, 0,
                               self.dq1, self.dq2
                              )

        self.k_q1, self.k_q2 = calculate_k(self.N_q1, self.N_q2,
                                           physical_system.dq1, 
                                           physical_system.dq2
                                          )

        self._initialize(rho_hat_initial)

    def initialize_magnetic_fields(self):
        
        if('initialize_B' in dir(self.initialize)):
            
            B1, B2, B3 = self.initialize.initialize_B(self.q1_center,
                                                      self.q2_center,
                                                      self.params
                                                     )

        # elif('initialize_A' in dir(self.initialize)):

        #     A1, A2, A3 = self.initialize.initialize_A(self.q1_center,
        #                                               self.q2_center,
        #                                               self.params
        #                                              )

        #     A1_hat = 2 * fft2(A1) / (self.N_q1 * self.N_q2)
        #     A2_hat = 2 * fft2(A2) / (self.N_q1 * self.N_q2)
        #     A3_hat = 2 * fft2(A3) / (self.N_q1 * self.N_q2)

        #     B1 =  ifft2(A3_hat * 1j * self.k_q2)
        #     B2 = -ifft2(A3_hat * 1j * self.k_q1)
        #     B3 =  ifft2(A2_hat * 1j * self.k_q1 - A1_hat * 1j * self.k_q2)

        elif('initialize_A3_B3' in dir(self.initialize)):

            A3 = self.initialize.initialize_A3_B3(self.q1_center,
                                                  self.q2_center,
                                                  self.params
                                                 )[0]

            A3_hat = fft2(A3)

            B1 =  ifft2(A3_hat * 1j * self.k_q2)
            B2 = -ifft2(A3_hat * 1j * self.k_q1)

            B3 = self.initialize.initialize_A3_B3(self.q1_center,
                                                  self.q2_center,
                                                  self.params
                                                 )[1]
            B3 = af.cast(B3, af.Dtype.c64)

        else:
            raise NotImplementedError('Initialization method for magnetic fields not valid/found')

        af.eval(B1, B2, B3)
        return(B1, B2, B3)

    def _initialize(self, rho_hat_initial):

        # If option is given as user-defined:
        if(self.params.fields_initialize == 'user-defined'):
            
            E1, E2, E3 = self.initialize.initialize_E(self.q1_center, self.q2_center, self.params)
            B1, B2, B3 = self.initialize_magnetic_fields()

            # Scaling Appropriately
            self.E1_hat = 2 * fft2(E1) / (self.N_q1 * self.N_q2)
            self.E2_hat = 2 * fft2(E2) / (self.N_q1 * self.N_q2)
            self.E3_hat = 2 * fft2(E3) / (self.N_q1 * self.N_q2)
            self.B1_hat = 2 * fft2(B1) / (self.N_q1 * self.N_q2)
            self.B2_hat = 2 * fft2(B2) / (self.N_q1 * self.N_q2)
            self.B3_hat = 2 * fft2(B3) / (self.N_q1 * self.N_q2)

        elif (self.params.fields_initialize == 'fft + user-defined magnetic fields'):
            compute_electrostatic_fields(self, rho_hat_initial)
            B1, B2, B3 = self.initialize_magnetic_fields()

            self.B1_hat = 2 * fft2(B1) / (self.N_q1 * self.N_q2)
            self.B2_hat = 2 * fft2(B2) / (self.N_q1 * self.N_q2)
            self.B3_hat = 2 * fft2(B3) / (self.N_q1 * self.N_q2)

        # Initializing EM fields using Poisson Equation:
        else:
            compute_electrostatic_fields(self, rho_hat_initial)

        self.fields_hat = af.join(0, af.join(0, self.E1_hat, self.E2_hat, self.E3_hat), 
                                  self.B1_hat, self.B2_hat, self.B3_hat
                                 )

        af.eval(self.fields_hat)
        return
    
    def get_fields(self):
        
        if(self.params.fields_type == 'electrodynamic'):
            
            self.E1_hat = self.fields_hat[0]
            self.E2_hat = self.fields_hat[1]
            self.E3_hat = self.fields_hat[2]
            
            self.B1_hat = self.fields_hat[3]
            self.B2_hat = self.fields_hat[4]
            self.B3_hat = self.fields_hat[5]

        return(self.E1_hat, self.E2_hat, self.E3_hat, 
               self.B1_hat, self.B2_hat, self.B3_hat
              )

    # Adding solver methods:
    compute_electrostatic_fields = compute_electrostatic_fields
