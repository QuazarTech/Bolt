#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af
# import pylab as pl

def compute_electrostatic_fields(self):
  """
  Computes the electrostatic fields by making use of the Poisson equation:
  div^2 phi = rho
  """
  rho_hat = self.physical_system.params.charge_electron * \
            2 * af.fft2(self.compute_moments('density'))/(self.N_q1 * self.N_q2)
  rho_hat = af.tile(rho_hat, 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
  phi_hat = self.physical_system.params.charge_electron * rho_hat/(self.k_q1**2 + self.k_q2**2)
  
  phi_hat[0, 0] = 0

  self.E1_hat = -phi_hat * (1j * self.k_q1)
  self.E2_hat = -phi_hat * (1j * self.k_q2)
  self.E3_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)
  
  self.B1_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)
  self.B2_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64) 
  self.B3_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)

  # E = 0.5 * self.N_q1 * self.N_q2 * af.ifft2(self.E1_hat)

  # pl.plot(E[:, :, 0])
  # pl.show()

  return