import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):
    rho  = 1 * q1**0
    T    = 2/3 * q1**0
    p1_b = 1 * q1**0

    m = params.mass_particle
    k = params.boltzmann_constant

    f = rho * af.sqrt(m / (2 * np.pi * k * T))**3 \
            * af.exp(-m * (p1-p1_b)**2 / (2 * k * T)) \
            * af.exp(-m * p2**2 / (2 * k * T)) \
            * af.exp(-m * p3**2 / (2 * k * T))

    return(f)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):
    rho  = 0.92307692 * q1**0
    T    = 0.63194444 * q1**0
    p1_b = 1.08333333 * q1**0

    m = params.mass_particle
    k = params.boltzmann_constant

    f = rho * af.sqrt(m / (2 * np.pi * k * T))**3 \
            * af.exp(-m * (p1-p1_b)**2 / (2 * k * T)) \
            * af.exp(-m * p2**2 / (2 * k * T)) \
            * af.exp(-m * p3**2 / (2 * k * T))

    return(f)
