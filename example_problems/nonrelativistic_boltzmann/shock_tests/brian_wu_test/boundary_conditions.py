import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    
    n = 1 * q1**0
    T = 1 * q1**0

    m = params.mass
    k = params.boltzmann_constant

    f = n * af.sqrt(m / (2 * np.pi * k * T))**3 \
          * af.exp(-m * p1**2 / (2 * k * T)) \
          * af.exp(-m * p2**2 / (2 * k * T)) \
          * af.exp(-m * p3**2 / (2 * k * T))

    return(f)

def E1_left(E1, t, q1, q2, params):
    return(0 * q1**0)

def E2_left(E2, t, q1, q2, params):
    return(0 * q1**0)

def E3_left(E3, t, q1, q2, params):
    return(0 * q1**0)

def B1_left(B1, t, q1, q2, params):
    return(0 * q1**0)

def B2_left(B2, t, q1, q2, params):
    return(params.B0 * 1 * q1**0)

def B3_left(B3, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def f_right(f, t, q1, q2, p1, p2, p3, params):
    
    n = 0.125 * q1**0
    T = 0.8   * q1**0

    m = params.mass
    k = params.boltzmann_constant

    f = n * af.sqrt(m / (2 * np.pi * k * T))**3 \
          * af.exp(-m * p1**2 / (2 * k * T)) \
          * af.exp(-m * p2**2 / (2 * k * T)) \
          * af.exp(-m * p3**2 / (2 * k * T))

    return(f)

def E1_right(E1, t, q1, q2, params):
    return(0 * q1**0)

def E2_right(E2, t, q1, q2, params):
    return(0 * q1**0)

def E3_right(E3, t, q1, q2, params):
    return(0 * q1**0)

def B1_right(B1, t, q1, q2, params):
    return(0 * q1**0)

def B2_right(B2, t, q1, q2, params):
    return(-params.B0 * 1 * q1**0)

def B3_right(B3, t, q1, q2, params):
    return(0 * q1**0)
