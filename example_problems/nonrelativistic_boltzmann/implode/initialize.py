"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background

    # Calculating the perturbed density:
    rho = af.select(q1+q2>0.15, q1**0, 0.125)
    T   = af.select(q1+q2>0.15, q1**0, 0.373)
 
    f = rho * (m / (2 * np.pi * T))**(3 / 2) \
            * af.exp(-p1**2 / (2 * T)) \
            * af.exp(-p2**2 / (2 * T)) \
            * af.exp(-p3**2 / (2 * T))   
    
    af.eval(f)
    return (f)
