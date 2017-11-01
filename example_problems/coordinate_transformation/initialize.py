"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    x = q1 * af.cos(q2)
    y = q1 * af.sin(q2)

    # Calculating the perturbed density:
    rho = af.exp(-(x**2 + (y - 1.5)**2)/0.25**2)
    f   = p1 * rho

    f[:] = 0

    f   = af.moddims(f, 32, 32, 70*70)
    rho = af.moddims(rho, 1, 1, 70*70)

    f[20, 20] = rho
    
    f = af.moddims(f, 32*32, 70, 70)

    af.eval(f)
    return (f)
