"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n = af.select(af.abs(q2)>0.25, q1**0, 2)

    # Random Numbers under to seed the instability:
    seeding_velocities = 0.01 * (af.randu(1, 1, q1.shape[2], q1.shape[3], 
                                          dtype = af.Dtype.f64
                                         ) - 0.5
                                )

    v1_bulk = af.select(af.abs(q2)>0.25, 
                        -0.5 - seeding_velocities,
                        +0.5 + seeding_velocities
                       )
       
    v2_bulk = seeding_velocities

    T = (2.5 / n)

    f = n * (m / (2 * np.pi * k * T))**(3 / 2) \
          * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
          * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) \
          * af.exp(-m * v3**2 / (2 * k * T))

    af.eval(f)
    return (f)
