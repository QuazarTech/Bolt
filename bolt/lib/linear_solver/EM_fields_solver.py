#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af


def compute_electrostatic_fields(self):
    """
    Computes the electrostatic fields by making use of the Poisson equation:
    div^2 phi = rho
    """
    rho_hat = 2 * af.fft2(self.compute_moments('density')) / \
              (self.N_q1 * self.N_q2)  # Scaling Appropriately
    phi_hat = self.physical_system.params.charge_electron * \
              rho_hat / (self.k_q1**2 + self.k_q2**2)

    phi_hat[0, 0] = 0

    # All field quantities apart from E1 and E2 are taken as zero:
    self.E1_hat = -phi_hat * (1j * self.k_q1)
    self.E2_hat = -phi_hat * (1j * self.k_q2)
    self.E3_hat = af.constant(0, self.N_q1, self.N_q2, dtype=af.Dtype.c64)

    self.B1_hat = af.constant(0, self.N_q1, self.N_q2, dtype=af.Dtype.c64)
    self.B2_hat = af.constant(0, self.N_q1, self.N_q2, dtype=af.Dtype.c64)
    self.B3_hat = af.constant(0, self.N_q1, self.N_q2, dtype=af.Dtype.c64)

    af.eval(self.E1_hat, self.E2_hat, self.E3_hat,
            self.B1_hat, self.B2_hat, self.B3_hat)
    
    del phi_hat, rho_hat; af.device_gc()
    
    return
