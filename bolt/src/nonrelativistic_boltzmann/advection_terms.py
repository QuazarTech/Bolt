#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.

The equation that we are solving is:

  df/dt + v_x * df/dq1 + v_y * df/dy 
+ e/m * (E + v X B)_x * df/dv_x 
+ e/m * (E + v X B)_y * df/dv_y 
+ e/m * (E + v X B)_y * df/dv_z = 0

In the solver framework this can be described using:

q1 = x  ; q2 = y
p1 = v1 = v_x; p2 = v2 = v_y; p3 = v3 = v_z

A_q1 = C_q1 = v_x = v1
A_q2 = C_q2 = v_y = v2

A_v1 = C_v1 = e/m * (E_x + v_y * B_z - v_z * B_y) = e/m * (E1 + v2 * B3 - v3 * B2)
A_v2 = C_v2 = e/m * (E_y + v_z * B_x - v_x * B_z) = e/m * (E2 + v3 * B1 - v1 * B3)
A_v3 = C_v3 = e/m * (E_z + v_x * B_y - v_y * B_x) = e/m * (E3 + v1 * B2 - v2 * B1)
"""

def A_q(f, t, q1, q2, v1, v2, v3, params):
    """
    Return the terms A_q1, A_q2.

    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    # A_q1[:, :N_s] = v1
    # A_q2[:, :N_s] = v2

    A_q1 = v1
    A_q2 = v2

    return (A_q1, A_q2)

# Conservative Advection terms in q-space:
# Used by the FVM solver:
def C_q(f, t, q1, q2, v1, v2, v3, params):
    """
    Return the terms C_q1, C_q2.

    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """
    # C_q1[:, :N_s] = v1
    # C_q2[:, :N_s] = v2
    
    C_q1 = v1
    C_q2 = v2

    return (C_q1, C_q2)

def A_p(f, t, q1, q2, v1, v2, v3,
        fields_solver,
        params
       ):
    """
    Return the terms A_v1, A_v2 and A_v3.

    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    fields_solver: The solver object whose method get_fields() is used to 
                   obtain the EM field quantities

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    e = params.charge
    m = params.mass

    E1, E2, E3, B1, B2, B3 = fields_solver.get_fields()

    A_p1 = (e/m) * (E1 + v2 * B3 - v3 * B2)
    A_p2 = (e/m) * (E2 + v3 * B1 - v1 * B3)
    A_p3 = (e/m) * (E1 + v1 * B2 - v2 * B1)

    return (A_p1, A_p2, A_p3)

# Conservative Advection terms in p-space:
# Used by the FVM solver:
def C_p(f, t, q1, q2, v1, v2, v3,
        fields_solver,
        params
       ):
    """
    Return the terms C_v1, C_v2 and C_v3.

    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    fields_solver: The solver object whose method get_fields() is used to 
                   obtain the EM field quantities

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    e = params.charge
    m = params.mass

    E1, E2, E3, B1, B2, B3 = fields_solver.get_fields()

    C_p1 = (e/m) * (E1 + v2 * B3 - v3 * B2)
    C_p2 = (e/m) * (E2 + v3 * B1 - v1 * B3)
    C_p3 = (e/m) * (E1 + v1 * B2 - v2 * B1)

    return (C_p1, C_p2, C_p3)
