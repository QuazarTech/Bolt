#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

def compute_moments(self, moment_name, f=None):
    """
    Used in computing the moments of the distribution function.
    The moment definitions which are passed to physical system
    are used in computing these moment quantities.

    Parameters
    ----------

    moments_name : str
                   Pass the moment name which needs to be computed.
                   It must be noted that this needs to be defined by the
                   user under moment_defs under src and passed to the 
                   physical_system object.
    
    f: af.Array
       Pass this argument as well when you want to compute the 
       moments of the input array and not the one stored by the state vector
       of the object.

    Examples
    --------
    
    >> solver.compute_moments('density')

    The above line will lookup the definition for 'density' under the dict
    moments_exponents, and moments_coefficients and calculate the same
    accordingly
    """
    N_g_p = self.N_ghost_p

    if(N_g_p != 0):
    
        p1 = af.flat(af.moddims(self.p1_center,
                                self.N_p1 + 2 * N_g_p,
                                self.N_p2 + 2 * N_g_p,
                                self.N_p3 + 2 * N_g_p,
                               )[N_g_p:-N_g_p, N_g_p:-N_g_p, N_g_p:-N_g_p]
                    ) 
        p2 = af.flat(af.moddims(self.p2_center,
                                self.N_p1 + 2 * N_g_p,
                                self.N_p2 + 2 * N_g_p,
                                self.N_p3 + 2 * N_g_p,
                               )[N_g_p:-N_g_p, N_g_p:-N_g_p, N_g_p:-N_g_p]
                    ) 
        p3 = af.flat(af.moddims(self.p3_center,
                                self.N_p1 + 2 * N_g_p,
                                self.N_p2 + 2 * N_g_p,
                                self.N_p3 + 2 * N_g_p,
                               )[N_g_p:-N_g_p, N_g_p:-N_g_p, N_g_p:-N_g_p]
                    ) 

    else:
        
        p1 = self.p1_center
        p2 = self.p2_center
        p3 = self.p3_center

    try:
        moment_exponents = \
            np.array(self.physical_system.moment_exponents[moment_name])
        moment_coeffs    = \
            np.array(self.physical_system.moment_coeffs[moment_name])

    except BaseException:
        raise KeyError('moment_name not defined under physical system')

    try:
        moment_variable = 1
        for i in range(moment_exponents.shape[0]):
            moment_variable *=   moment_coeffs[i, 0] \
                               * p1**(moment_exponents[i, 0]) \
                               + moment_coeffs[i, 1] \
                               * p2**(moment_exponents[i, 1]) \
                               + moment_coeffs[i, 2] \
                               * p3**(moment_exponents[i, 2])

    except BaseException:
        moment_variable =   moment_coeffs[0] * p1**(moment_exponents[0]) \
                          + moment_coeffs[1] * p2**(moment_exponents[1]) \
                          + moment_coeffs[2] * p3**(moment_exponents[2])

    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    multiply = lambda a, b:a*b

    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    if(f is None):
        
        N_q1 = self.f.shape[1]
        N_q2 = self.f.shape[2]

        if(N_g_p != 0):
            f = af.moddims(self._convert_to_p_expanded(self.f)[N_g_p:-N_g_p, 
                                                               N_g_p:-N_g_p,
                                                               N_g_p:-N_g_p
                                                              ],
                           self.N_p1 * self.N_p2 * self.N_p3, N_q1, N_q2
                          )

        else:
            f = self.f
    
    else:
        N_q1 = f.shape[1]
        N_q2 = f.shape[2]

        if(N_g_p != 0):
            f = af.moddims(self._convert_to_p_expanded(f)[N_g_p:-N_g_p, 
                                                          N_g_p:-N_g_p,
                                                          N_g_p:-N_g_p
                                                         ],
                           self.N_p1 * self.N_p2 * self.N_p3, N_q1, N_q2
                          )

    moment   =   af.sum(af.broadcast(multiply, f, moment_variable), 0) \
               * self.dp3 * self.dp2 * self.dp1

    af.eval(moment)
    return (moment)
