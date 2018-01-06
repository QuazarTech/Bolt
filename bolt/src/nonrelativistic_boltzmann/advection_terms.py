"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.

The equation that we are solving is:

df/dt + v_x * df/dq1 + v_y * df/dy + (E + v X B)_x * df/dv_x + (E + v X B)_y * df/dv_y + (E + v X B)_y * df/dv_z = 0

In the solver framework this can be described using:

q1 = x  ; q2 = y
p1 = v_x; p2 = v_y; p3 = v_z

A_q1 = C_q1 = v_x = p1
A_q2 = C_q2 = v_y = p2

A_p1 = C_p1 = q/m * (E_x + v_y * B_z - v_z * B_y) = q/m * (E1 + p2 * B3 - p3 * B2)
A_p2 = C_p2 = q/m * (E_y + v_z * B_x - v_x * B_z) = q/m * (E2 + p3 * B1 - p1 * B3)
A_p3 = C_p3 = q/m * (E_z + v_x * B_y - v_y * B_x) = q/m * (E3 + p1 * B2 - p2 * B1)
"""

def A_q(q1, q2, p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    return (p1, p2)

# Conservative Advection terms in q-space:
# Used by the FVM solver:
def C_q(q1, q2, p1, p2, p3, params):
    """Return the terms C_q1, C_q2."""
    return (p1, p2)

def A_p(q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """Return the terms A_p1, A_p2 and A_p3."""
    F1 =   (params.charge / params.mass) \
         * (E1 + p2 * B3 - p3 * B2)
    F2 =   (params.charge / params.mass) \
         * (E2 + p3 * B1 - p1 * B3)
    F3 =   (params.charge / params.mass) \
         * (E3 + p1 * B2 - p2 * B1)

    return (F1, F2, F3)

# Conservative Advection terms in p-space:
def C_p(q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """Return the terms C_p1, C_p2 and C_p3."""
    F1 =   (params.charge / params.mass) \
         * (E1 + p2 * B3 - p3 * B2)
    F2 =   (params.charge / params.mass) \
         * (E2 + p3 * B1 - p1 * B3)
    F3 =   (params.charge / params.mass) \
         * (E3 + p1 * B2 - p2 * B1)

    return (F1, F2, F3)
