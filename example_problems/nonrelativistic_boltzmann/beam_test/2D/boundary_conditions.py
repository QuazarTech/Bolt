import numpy as np
import arrayfire as af

in_q1 = 'dirichlet'
in_q2 = 'dirichlet'

@af.broadcast
def f_left(q1, q2, p1, p2, p3, params):
    f    = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    
    f           = af.moddims(f, 148*148, 4, 4)
    f_activated = f.copy()
    
    f_activated[:, 2, 2] = 1
    
    f           = af.moddims(f, 148, 148, 4*4)
    f_activated = af.moddims(f_activated, 148, 148, 4*4)
    
    f[:, 60:87] = f_activated[:, 60:87]
    return(f)

@af.broadcast
def f_right(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    return(f)

@af.broadcast
def f_bot(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    return(f)

@af.broadcast
def f_top(q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    return(f)
    