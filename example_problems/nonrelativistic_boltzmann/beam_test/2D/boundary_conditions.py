import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    f    = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    
    f           = af.moddims(f, 4, 4, 1038*178)
    f_activated = f.copy()
    
    f_activated[3, 2] = 1
    f_activated[3, 1] = 1
    
    f           = af.moddims(f, 4*4, 1, 1038, 178)
    f_activated = af.moddims(f_activated, 4*4, 1, 1038, 178)
    
    f[:, :, :, 71:105] = f_activated[:, :, :, 71:105]
    return(f)

@af.broadcast
def f_right(f, t, q1, q2, p1, p2, p3, params):
    f = q1**0 * np.sqrt(1 / (4 * np.pi)) * af.exp(-p1**2 / 4)
    f[:] = 0
    return(f)
