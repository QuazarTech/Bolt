import numpy as np
import arrayfire as af
import params

in_q1_left  = 'dirichlet'
in_q1_right = 'mirror'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, v1, v2, v3, params):
    k            = params.boltzmann_constant
    n_left       = params.n_left
    v1_bulk_left = params.v1_bulk_left
    T_left       = params.T_left

    f =   q1**0 * n_left * (params.mass / (2 * np.pi * k * T_left))**(3 / 2) \
        * af.exp(-params.mass * (v1 - v1_bulk_left)**2 / (2 * k * T_left)) \
        * af.exp(-params.mass * v2**2 / (2 * k * T_left)) \
        * af.exp(-params.mass * v3**2 / (2 * k * T_left))

    return(f)

@af.broadcast
def E1_left(E1, t, q1, q2, params):
    return(0 * params.E0 * q1**0)

@af.broadcast
def E2_left(E2, t, q1, q2, params):
    return(0 * params.E0 * q1**0)

@af.broadcast
def E3_left(E3, t, q1, q2, params):
    return(0 * params.E0 * q1**0)

@af.broadcast
def B1_left(B1, t, q1, q2, params):
    return(params.B1 * q1**0)

@af.broadcast
def B2_left(B2, t, q1, q2, params):
    return(0 * q1**0)

@af.broadcast
def B3_left(B3, t, q1, q2, params):
    return(0 * q1**0)
