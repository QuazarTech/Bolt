#import numpy as np
import arrayfire as af

#@af.broadcast
def A_q(q1, q2, p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    
    A_q1, A_q2 = params.vel_band

    return (A_q1, A_q2)

def C_q(q1, q2, p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    
    C_q1, C_q2 = params.vel_band

    return (C_q1, C_q2)

# This can then be called inside A_p if needed:
# F1 = (params.char....)(E1 + ....) + T1(q1, q2, p1, p2, p3)

def A_p(q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """Return the terms A_p1, A_p2 and A_p3."""
    e = params.charge_electron
    c = params.speed_of_light

#    dp1_dt = e*(E1 + (p2*B3 - p3*B2) / c) # p1 = hcross * k1
#    dp2_dt = e*(E2 + (p3*B1 - p1*B3) / c) # p2 = hcross * k2
#    dp3_dt = e*(E3 + (p1*B2 - p2*B1) / c) # p3 = hcross * k3

    dp1_dt = -e*E1 + 0.*p1**0.
    dp2_dt = -e*E2 + 0.*p1**0.
    dp3_dt = 0. + 0.*p1**0.

    #dp1_dt = 1e-5*e*0.5*(-af.tanh(100.*(q1 - 0.9)) - af.tanh(100.*(q1 - 0.1)) )
#    amplitude   = 1e-3
#    E1_analytic = amplitude * -(q1 - 5.)
#    dp1_dt      = E1_analytic

    #E2_analytic = amplitude * -(q2 - 5.)
    #dp2_dt      = E2_analytic
    
    return (dp1_dt, dp2_dt, dp3_dt)
