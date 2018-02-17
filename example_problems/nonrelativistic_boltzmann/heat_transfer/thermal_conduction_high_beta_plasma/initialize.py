"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
import domain

# Initial conditions of Komarov et al.
def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
    k = params.boltzmann_constant

    T_l = params.temperature_left
    T_r = params.temperature_right
    L   = domain.q1_end - domain.q1_start

    T = T_l + (T_r - T_l) / L * q1
    n = params.pressure / T
    
    f = n * (m / (2 * np.pi * k * T))**(3 / 2) \
          * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
          * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
          * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T))

    af.eval(f)
    return (f)

# Initial conditions of Roberg-Clark et al.
# Unsure. Ignore the following
def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
    k = params.boltzmann_constant

    T_l = params.temperature_left
    T_r = params.temperature_right
    L   = domain.q1_end - domain.q1_start

    T = T_l + (T_r - T_l) / L * q1
    n = params.pressure / T
    

    f = n * (m / (np.pi * k))**(3 / 2) \
          * (  af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T))
             +     af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T))
                 * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T))
                 / (1 + af.erf(p1_bulk))
            )
          
    af.eval(f)
    return (f)

def initialize_B(q1, q2, params):

    B1 = 1 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    return(B1, B2, B3)
