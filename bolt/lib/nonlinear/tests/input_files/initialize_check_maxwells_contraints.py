import numpy as np
import arrayfire as af

def initialize_A3_B3(q1, q2, params):
    A3 = af.sin(2 * np.pi * q1 + 4 * np.pi * q2)
    B3 = 1e-5 * q1**0

    return(A3, B3)
