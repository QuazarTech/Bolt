"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np


def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    # Calculating the perturbed density:
    rho = 1 + (  pert_real * af.cos(k_q1 * q1 + k_q2 * q2)
               - pert_imag * af.sin(k_q1 * q1 + k_q2 * q2)
              )
    
    n_p = 0.9/np.sqrt(2*np.pi)
    n_b = 0.2/np.sqrt(2*np.pi)
  
    f =   rho \
        * (  n_p * af.exp(-0.5*p1**2) 
           + n_b * af.exp(-0.5*((p1 - 4.5)/0.5)**2)
          )

    af.eval(f)
    return (f)
