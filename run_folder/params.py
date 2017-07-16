import numpy as np

fields_initialize = 'electrostatic'
timestepper       = 'RK6'

p_dim = 1  

mass_particle      = 1
boltzmann_constant = 1
charge_electron    = -10

rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0
p3_bulk_background = 0

pert_real = 0.01
pert_imag = 0

k_q1 = 2*np.pi
k_q2 = 0

def tau(q1, q2, p1, p2, p3):
  return(0.01)