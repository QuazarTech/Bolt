#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""Contains the function which returns the Source/Sink term."""

import numpy as np
import arrayfire as af

from bolt.lib.nonlinear.utils.broadcasted_primitive_operations import multiply

# Using af.broadcast, since v1, v2, v3 are of size (1, 1, Nv1*Nv2*Nv3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params):
    """
    Return the Local MB distribution.
    Parameters:
    -----------
    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)
    
    n: Computed density at the current state.
       shape:(1, N_s, N_q1, N_q2)

    T: Computed temperature at the current state.
       shape:(1, N_s, N_q1, N_q2)

    v1_bulk: Computed bulk velocity in the q1 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    v2_bulk: Computed bulk velocity in the q2 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    v3_bulk: Computed bulk velocity in the q3 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """
    m = params.mass
    k = params.boltzmann_constant

    if (params.p_dim == 3):
        f0 = n * (m / (2 * np.pi * k * T))**(3 / 2)  \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):
        f0 = n * (m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T))

    else:
        f0 = n * af.sqrt(m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T))

    af.eval(f0)
    return (f0)

def BGK(f, t, q1, q2, v1, v2, v3, moments, params, flag = False):
    """
    Return BGK operator -(f-f0)/tau.

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
    
    flag: Toggle used for evaluating tau = 0 cases need to be evaluated. When set to True, this
          function is made to return f0, thus setting f = f0 wherever tau = 0
    """
    n = moments('density', f)
    m = params.mass

    # Floor used to avoid 0/0 limit:
    eps = 1e-30

    v1_bulk = moments('mom_v1_bulk', f) / (n + eps)
    v2_bulk = moments('mom_v2_bulk', f) / (n + eps)
    v3_bulk = moments('mom_v3_bulk', f) / (n + eps)

    T = (1 / params.p_dim) * (  2 * multiply(moments('energy', f), m)
                                  - multiply(n, m) * v1_bulk**2
                                  - multiply(n, m) * v2_bulk**2
                                  - multiply(n, m) * v3_bulk**2
                             ) / (n + eps) + eps

    f_MB = f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params)
    tau  = params.tau(q1, q2, v1, v2, v3)

    if(flag == False):

        C_f = -(f - f_MB) / tau
        return(C_f)

    # Accounting for purely collisional cases(f = f0):
    else:
        return(f_MB)
