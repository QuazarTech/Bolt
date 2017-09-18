#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def compute_electrostatic_fields(self):
    """
    Computes the electrostatic fields by making use of the
    Poisson equation: div^2 phi = rho
    """
    rho_hat = 2 * af.fft2(self.compute_moments('density')) \
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

    af.eval(self.E1_hat, self.E2_hat)
    return
