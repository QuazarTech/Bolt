"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
    k = params.boltzmann_constant

    n = params.density_background

    beta_1 = params.beta_1
    beta_2 = params.beta_2

    f = n * (1 / np.pi)**(3/2) * (1 / np.sqrt(beta_1) * beta_2) \
          * af.exp(-p1**2 / beta_1) \
          * af.exp(-p2**2 / beta_2) \

    af.eval(f)
    return (f)
