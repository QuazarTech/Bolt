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

        rho_hat = 2 * af.fft2(self.compute_moments('density', f=f, f_hat=f_hat)) \
                    / (self.N_q1 * self.N_q2) # Scaling Appropriately

        # Defining lambda functions to perform broadcasting operations:
        # This is done using af.broadcast, which allows us to perform 
        # batched operations when operating on arrays of different sizes:
        divide   = lambda a, b:a/b
        multiply = lambda a, b:a*b

        # af.broadcast(function, *args) performs batched operations on
        # function(*args):
        phi_hat = self.physical_system.params.charge_electron * \
                  af.broadcast(divide, rho_hat, 
                               (self.k_q1**2 + self.k_q2**2)
                              )

        # Setting the background electric potential to zero:
        phi_hat[0, 0] = 0

        # Tiling to make phi_hat of positionsExpanded form:
        phi_hat = af.tile(phi_hat, 1, 1, self.N_p1 * self.N_p2 * self.N_p3)

        self.E1_hat = af.broadcast(multiply, -phi_hat, (1j * self.k_q1))
        self.E2_hat = af.broadcast(multiply, -phi_hat, (1j * self.k_q2))
        self.E3_hat = 0 * self.E1_hat 
        
        self.B1_hat = 0 * self.E1_hat
        self.B2_hat = 0 * self.E1_hat
        self.B3_hat = 0 * self.E1_hat

        af.eval(self.E1_hat, self.E2_hat, self.E3_hat,
                self.E1_hat, self.E2_hat, self.E3_hat
               )

    return
