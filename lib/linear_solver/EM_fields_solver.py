#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def compute_electrostatic_fields(self):
  """
  Computes the electrostatic fields by making use of the Poisson equation:
  div^2 phi = rho
  """
  rho_hat = 2 * af.fft2(self.compute_moments('density'))/(self.N_q1 * self.N_q2)
  phi_hat = self.physical_system.params.charge_electron * rho_hat/(self.k_q1**2 + self.k_q2**2)
  
  phi_hat[0, 0] = 0

  self.E_x_hat = -phi_hat * (1j * self.k_q1)
  self.E_y_hat = -phi_hat * (1j * self.k_q2)
  self.E_z_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)
  
  self.B_x_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)
  self.B_y_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64) 
  self.B_z_hat = af.constant(0, self.N_q1, self.N_q2, self.N_p1*self.N_p2*self.N_p3, dtype = af.Dtype.c64)

  return