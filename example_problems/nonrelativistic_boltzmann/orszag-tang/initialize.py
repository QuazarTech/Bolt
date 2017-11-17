"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_E(q1, q2, params):
    
    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    Bx = -B0Sin(2πy) and By = B0Sin(4πx)
    
    B1 = -(1/np.sqrt(4 * np.pi)) * af.sin(2 * np.pi * q2)
    B2 =  (1/np.sqrt(4 * np.pi)) * af.sin(4 * np.pi * q1)
    B3 = 0 * q1**0

    return(B1, B2, B3)

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = 25/(36 * np.pi)
    Vx = - Sin(2πy) and Vy = Sin(2πx)

    p1_bulk = - af.sin(2 * np.pi * q2)
    p2_bulk =   af.sin(2 * np.pi * q1)

    T = ((5 / (12 * np.pi)) / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p3)**2 / (2 * k * T))

    af.eval(f)
    return (f)
