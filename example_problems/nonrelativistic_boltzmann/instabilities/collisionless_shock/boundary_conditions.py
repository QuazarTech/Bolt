import arrayfire as af
import numpy as np

in_q1_left  = 'dirichlet'
in_q1_right = 'mirror'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, v1, v2, v3, params):
    n       = params.density
    v1_bulk = params.v1_bulk
    beta    = params.beta

    f =   q1**0 * (n / (np.pi * beta)**(3 / 2)) \
        * af.exp(-(v1 - v1_bulk)**2 / beta) \
        * af.exp(-v2**2 / beta) \
        * af.exp(-v3**2 / beta)

    return(f)

@af.broadcast
def E1_left(E1, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def E2_left(E2, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def E3_left(E3, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def B1_left(B1, t, q1, q2, params):
    return(1 * q1**0)

@af.broadcast
def B2_left(B2, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def B3_left(B3, t, q1, q2, params):
    return(0 * q1**0)
