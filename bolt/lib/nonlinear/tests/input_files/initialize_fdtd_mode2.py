import numpy as np
import arrayfire as af

def initialize_A_E3(q1, q2, params):

    A  = -1.5 * np.sqrt(5/9) * af.cos(2 * np.pi * q1 + 4 * np.pi * q2)
    E3 = 0 * q1**0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_A3_B3(q1, q2, params):

    dt = params.dt

    A3 = 0 * q1**0
    B3 = -5 * np.pi * af.cos(  2 * np.pi * (q1 - 0.5 * np.sqrt(5 / 9) * dt)
                             + 4 * np.pi * (q2 - 0.5 * np.sqrt(5 / 9) * dt)
                            )

    af.eval(A3, B3)
    return(A3, B3)
