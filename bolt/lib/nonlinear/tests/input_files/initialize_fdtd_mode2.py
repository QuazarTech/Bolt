import numpy as np
import arrayfire as af

def initialize_E(q1, q2, params):

    E1 = 6 * np.pi * np.sqrt(5/9) * af.cos(2 * np.pi * q1 + 4 * np.pi * q2) 
    E2 = -3 * np.pi * np.sqrt(5/9) * af.cos(2 * np.pi * q1 + 4 * np.pi * q2) 
    E3 = 0 * q1**0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt

    B1 = 0 * q1**0
    B2 = 0 * q1**0
    B3 = -5 * np.pi * af.cos(  2 * np.pi * (q1 - 0.5 * np.sqrt(5 / 9) * dt)
                             + 4 * np.pi * (q2 - 0.5 * np.sqrt(5 / 9) * dt)
                            )

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
