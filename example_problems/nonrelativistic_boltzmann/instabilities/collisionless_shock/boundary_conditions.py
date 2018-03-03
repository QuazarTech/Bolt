import numpy as np
import arrayfire as af
import params

in_q1_left  = 'dirichlet'
in_q1_right = 'mirror'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    f    = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[2] = 1
    
    return(f)

@af.broadcast
def f_left(f, t, q1, q2, v1, v2, v3, params):
    k            = params.boltzmann_constant
    n_left       = params.n_left
    v1_bulk_left = params.v1_bulk_left
    T_left       = params.T_left
    plasma_beta  = params.plasma_beta

    f =   q1**0 * n_left * (params.mass / (2 * np.pi * k * T_left))**(3 / 2) \
        * af.exp(-(v1 - v1_bulk_left)**2 / (2 * k * T_left)) \
        * af.exp(-v2**2 / (2 * k * T_left)) \
        * af.exp(-v3**2 / (2 * k * T_left))

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
    return(  np.sqrt(2 * params.n_left * params.T_left / params.plasma_beta) 
           * params.B0 * q1**0
          )

@af.broadcast
def B2_left(B2, t, q1, q2, params):
    return(0 * params.B0 * q1**0)

@af.broadcast
def B3_left(B3, t, q1, q2, params):
    return(0 * params.B0 * q1**0)
