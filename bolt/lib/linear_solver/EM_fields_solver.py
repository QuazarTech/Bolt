#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def compute_electrostatic_fields(self, f=None, f_hat=None):
    """
    Computes the electrostatic fields by making use of the
    Poisson equation: div^2 phi = rho
    """

    if(self.single_mode_evolution == True):
        
        # Intializing for the electrostatic Case:
        delta_rho_hat = self.compute_moments('density', f)
        delta_phi_hat =   self.physical_system.params.charge_electron \
                        * delta_rho_hat/(  self.physical_system.params.k_q1**2 
                                         + self.physical_system.params.k_q2**2
                                        )

        self.delta_E1_hat = -delta_phi_hat * (1j * self.physical_system.params.k_q1)
        self.delta_E2_hat = -delta_phi_hat * (1j * self.physical_system.params.k_q2)
        self.delta_E3_hat = 0

        self.delta_B1_hat = 0 
        self.delta_B2_hat = 0 
        self.delta_B3_hat = 0 

    else:

        for i in range(self.N_species):
            n       = self.compute_moments('density', i, f=f, f_hat=f_hat)
            rho_hat = 2 * af.fft2(n - af.mean(n)) \
                        / (self.N_q1 * self.N_q2)

            if(i == 0):
                phi_hat = self.physical_system.params.charge[i] * \
                          rho_hat / (self.k_q1**2 + self.k_q2**2)
            else:
                phi_hat += self.physical_system.params.charge[i] * \
                           rho_hat / (self.k_q1**2 + self.k_q2**2)

        # Setting the background electric potential to zero:
        phi_hat[0, 0] = 0

        self.E1_hat = -phi_hat * 1j * self.k_q1
        self.E2_hat = -phi_hat * 1j * self.k_q2
        self.E3_hat = 0 * self.E1_hat 
        
        self.B1_hat = 0 * self.E1_hat
        self.B2_hat = 0 * self.E1_hat
        self.B3_hat = 0 * self.E1_hat

        af.eval(self.E1_hat, self.E2_hat, self.E3_hat,
                self.B1_hat, self.B2_hat, self.B3_hat
               )

    return
