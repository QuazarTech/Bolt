import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    rho = 1 * q1**0
    T   = 1 * q1**0

    m     = params.mass
    k     = params.boltzmann_constant

    f = rho * af.sqrt(m / (2 * np.pi * k * T))**3 \
            * af.exp(-m * p1**2 / (2 * k * T)) \
            * af.exp(-m * p2**2 / (2 * k * T)) \
            * af.exp(-m * p3**2 / (2 * k * T))

    return(f)

@af.broadcast
def f_right(f, t, q1, q2, p1, p2, p3, params):
    rho = 0.125 * q1**0
    T   = 0.8   * q1**0

    m     = params.mass
    k     = params.boltzmann_constant

    f = rho * af.sqrt(m / (2 * np.pi * k * T))**3 \
            * af.exp(-m * p1**2 / (2 * k * T)) \
            * af.exp(-m * p2**2 / (2 * k * T)) \
            * af.exp(-m * p3**2 / (2 * k * T))

    return(f)

