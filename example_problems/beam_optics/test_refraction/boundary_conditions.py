import numpy as np
import arrayfire as af

in_q1 = 'dirichlet'
in_q2 = 'mirror'

@af.broadcast
def f_left(q1, q2, p1, p2, p3, params):
    f    = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    
    f           = af.moddims(f, 49*49, 4, 4)
    f_activated = f.copy()
    
    f_activated[:, 3, 2] = 1
    
    f           = af.moddims(f, 49, 49, 4*4)
    f_activated = af.moddims(f_activated, 49, 49, 4*4)
    
    f[:, 19:28] = f_activated[:, 19:28]
    return(f)

@af.broadcast
def f_right(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    return(f)
