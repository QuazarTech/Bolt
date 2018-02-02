"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):
    f = 0.1 * af.select(af.abs(p1)<5.0, p1**0, 0)
    af.eval(f)
    return (f)
