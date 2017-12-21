import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror'

@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = 0.*params.T + 3e-4*4.

    fermi_dirac = (1./(af.exp( (E_upper - 0.*0.01)/(k*T) ) + 1.)
                  )
    
    q2_contact_start = 2.5; q2_contact_end = 7.5
    cond = ((q2 >= q2_contact_start) & \
            (q2 <= q2_contact_end) \
           )
    
    f_left = cond*fermi_dirac + (1 - cond)*f

    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = 0.*params.T + 3e-4*4.

    fermi_dirac = (1./(af.exp( (E_upper - 0.*0.01)/(k*T) ) + 1.)
                  )
    
    q2_contact_start = 2.5; q2_contact_end = 7.5
    cond = ((q2 >= q2_contact_start) & \
            (q2 <= q2_contact_end) \
           )
    
    f_right = cond*fermi_dirac + (1 - cond)*f

    af.eval(f_right)
    return(f_right)
