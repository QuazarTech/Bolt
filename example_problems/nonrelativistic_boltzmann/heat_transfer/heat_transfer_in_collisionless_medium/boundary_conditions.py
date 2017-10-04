import numpy as np
import arrayfire as af

in_q1 = 'dirichlet'
in_q2 = 'periodic'

@af.broadcast
def f_left(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    return(f)

@af.broadcast
def f_right(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    return(f)
    