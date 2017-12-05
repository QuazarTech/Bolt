import arrayfire as af
import domain

in_q1 = 'mirror'
in_q2 = 'mirror'

@af.broadcast
def f_left(q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = 0.*params.T + 3e-4*4.
    
    f = (1./(af.exp( (E_upper - 0.01)/(k*T) 
                  ) + 1.
           ))

    af.eval(f)
    return(f)

@af.broadcast
def f_right(q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = 0.*params.T + 3e-4*4.

    f = (1./(af.exp( (E_upper - 0.01)/(k*T) 
                  ) + 1.
           ))

    af.eval(f)
    return(f)
