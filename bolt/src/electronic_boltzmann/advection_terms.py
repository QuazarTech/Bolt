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
    B3_mean = params.B3_mean

    v1, v2 = params.vel_band

    dp1_dt = -e*(E1 + v2*B3_mean/c) # p1 = hcross * k1
    dp2_dt = -e*(E2 - v1*B3_mean/c) # p2 = hcross * k2
    dp3_dt = 0.*p1

    return (dp1_dt, dp2_dt, dp3_dt)
