import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):
    rho = 1 * q1**0
    T   = 1 * q1**0

    m     = params.mass_particle
    k     = params.boltzmann_constant
    gamma = params.gamma

    f = rho * af.sqrt(3 * m * (gamma - 1) / (4 * np.pi * k * T))**3 \
            * af.exp(-3 * m * (gamma - 1) * p1**2 / (4 * k * T)) \
            * af.exp(-3 * m * (gamma - 1) * p2**2 / (4 * k * T)) \
            * af.exp(-3 * m * (gamma - 1) * p3**2 / (4 * k * T))

    return(f)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):
    rho = 0.125 * q1**0
    T   = 0.8   * q1**0

    m     = params.mass_particle
    k     = params.boltzmann_constant
    gamma = params.gamma

    f = rho * af.sqrt(3 * m * (gamma - 1) / (4 * np.pi * k * T))**3 \
            * af.exp(-3 * m * (gamma - 1) * p1**2 / (4 * k * T)) \
            * af.exp(-3 * m * (gamma - 1) * p2**2 / (4 * k * T)) \
            * af.exp(-3 * m * (gamma - 1) * p3**2 / (4 * k * T))

    return(f)
