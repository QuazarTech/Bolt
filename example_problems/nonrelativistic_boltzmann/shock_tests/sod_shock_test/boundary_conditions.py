in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'


def f_left(f, q1, q2, p1, p2, p3, params):
    rho = 1
    T   = 1

    m = params.mass_particle
    k = params.boltzmann_constant

    f = rho * np.sqrt(m / (2 * np.pi * k * T)) \
            * af.exp(-m * p1**2 / (2 * k * T))

    return(f)

def f_right(f, q1, q2, p1, p2, p3, params):
    rho = 0.125
    T   = 0.8

    m = params.mass_particle
    k = params.boltzmann_constant

    f = rho * np.sqrt(m / (2 * np.pi * k * T)) \
            * af.exp(-m * p1**2 / (2 * k * T))

    return(f)
