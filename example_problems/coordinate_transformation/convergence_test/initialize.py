"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
import domain
import params

def initialize_f(r, theta, rdot, thetadot, phidot, params):

    # Using a transformation to get the coordinates in the form used in regular cartesian coordinates:
    q1 = r * af.cos(theta)
    q2 = r * af.sin(theta)

    p1 = rdot * af.cos(theta) - r * af.sin(theta) * thetadot
    p2 = rdot * af.sin(theta) + r * af.cos(theta) * thetadot

    q10 = params.q10
    q20 = params.q20

    p10 = params.p10
    p20 = params.p20

    sigma_q = params.sigma_q
    sigma_p = params.sigma_p

    q_profile = (1 / sigma_q**2 / (2 * np.pi)) * \
                af.exp(-0.5 * ((q1 - q10)**2 + (q2 - q20)**2) / sigma_q**2)
    p_profile = (1 / sigma_p**2 / (2 * np.pi)) * \
                af.exp(-0.5 * ((p1 - p10)**2 + (p2 - p20)**2) / sigma_p**2)
    
    f = q_profile * p_profile

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1;
    E2 = 0 * q1;
    E3 = 0 * q1;

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0 * q1
    B2 = 0 * q1
    B3 = 0 * q1

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
