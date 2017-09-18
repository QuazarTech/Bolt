"""Contains the function which returns the Source/Sink term."""


import numpy as np
import arrayfire as af

# Using af.broadcast, since p1, p2, p3 are of size (1, 1, Np1*Np2*Np3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params):
    """Return the Local MB distribution."""
    m = params.mass_particle
    k = params.boltzmann_constant

    if (params.p_dim == 3):
        f0 = n * (m / (2 * np.pi * k * T))**(3 / 2)  \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):
        f0 = n * (m / (2 * np.pi * k * T)) \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T))

    else:
        f0 = n * af.sqrt(m / (2 * np.pi * k * T)) \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T))

    af.eval(f0)
    return (f0)


def BGK(f, q1, q2, p1, p2, p3, moments, params):
    """Return BGK operator -(f-f0)/tau."""
    n = moments('density')

    p1_bulk = moments('mom_p1_bulk') / n
    p2_bulk = moments('mom_p2_bulk') / n
    p3_bulk = moments('mom_p3_bulk') / n

    T =   (1 / params.p_dim) \
        * (  moments('energy') 
           - n * p1_bulk**2
           - n * p2_bulk**2
           - n * p3_bulk**2
          ) / n

    C_f = -(  f
            - f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params)
           ) / params.tau(q1, q2, p1, p2, p3)

    af.eval(C_f)
    return(C_f)
