import numpy as np
import arrayfire as af

in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    f    = 0 * q1**0 * p1**0
    
    f           = af.moddims(f, 4, 1, q1.elements())
    f_activated = f.copy()
    
    f_activated[3] = 1
    
    N = int(round(np.sqrt(q1.elements())))

    f           = af.moddims(f, 4*1, 1, N, N)
    f_activated = af.moddims(f_activated, 4*1, 1, N, N)
    
    N_lower = int(round(0.4 * N))
    N_upper = int(round(0.6 * N))

    print(N_lower)
    print(N_upper)

    f[:, :, :, N_lower:N_upper] = (af.exp(-250 * (q2 - 0.5)**2) * f_activated)[:, :, :, N_lower:N_upper]
    return(f)

@af.broadcast
def f_right(f, t, q1, q2, p1, p2, p3, params):
    f = 0 * q1**0 * p1**0
    return(f)
