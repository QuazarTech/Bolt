#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.

The equation that we are solving is:

df/dt + v_x * df/dq1 + v_y * df/dy + (E + v X B)_x * df/dv_x + (E + v X B)_y * df/dv_y + (E + v X B)_y * df/dv_z = 0

In the solver framework this can be described using:

q1 = x  ; q2 = y
v1 = v_x; v2 = v_y; v3 = v_z

A_q1 = C_q1 = v_x = v1
A_q2 = C_q2 = v_y = v2

A_v1 = C_v1 = q/m * (E_x + v_y * B_z - v_z * B_y) = q/m * (E1 + v2 * B3 - v3 * B2)
A_v2 = C_v2 = q/m * (E_y + v_z * B_x - v_x * B_z) = q/m * (E2 + v3 * B1 - v1 * B3)
A_v3 = C_v3 = q/m * (E_z + v_x * B_y - v_y * B_x) = q/m * (E3 + v1 * B2 - v2 * B1)
"""

def A_q(f, t, q1, q2, v1, v2, v3, params):
    """Return the terms A_q1, A_q2."""
    A_q1 = v1
    A_q2 = v2

    return (A_q1, A_q2)

# Conservative Advection terms in q-space:
# Used by the FVM solver:
def C_q(f, t, q1, q2, v1, v2, v3, params):
    """Return the terms C_q1, C_q2."""
    C_q1 = v1
    C_q2 = v2

    return (C_q1, C_q2)

def A_p(f, t, q1, q2, v1, v2, v3,
        fields_solver,
        params
       ):
    """Return the terms A_v1, A_v2 and A_v3."""
    q = params.charge
    m = params.mass

    E1, E2, E3, B1, B2, B3 = fields_solver.get_fields()

    A_p1 = (q/m) * (E1 + v2 * B3 - v3 * B2)
    A_p2 = (q/m) * (E2 + v3 * B1 - v1 * B3)
    A_p3 = (q/m) * (E1 + v1 * B2 - v2 * B1)

    return (A_p1, A_p2, A_p3)

# Conservative Advection terms in p-space:
# Used by the FVM solver:
def C_p(f, t, q1, q2, v1, v2, v3,
        fields_solver,
        params
       ):
    """Return the terms C_v1, C_v2 and C_v3."""
    q = params.charge
    m = params.mass

    E1, E2, E3, B1, B2, B3 = fields_solver.get_fields()

    C_p1 = (q/m) * (E1 + v2 * B3 - v3 * B2)
    C_p2 = (q/m) * (E2 + v3 * B1 - v1 * B3)
    C_p3 = (q/m) * (E1 + v1 * B2 - v2 * B1)

    return (C_p1, C_p2, C_p3)
