#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np


def compute_moments(self, moment_name):
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

    Examples
    --------
    
    >> solver.compute_moments('density')

    The above line will lookup the definition for 'density' under the dict
    moments_exponents, and moments_coefficients and calculate the same
    accordingly
    """
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
                               * self.p1**(moment_exponents[i, 0]) \
                               + moment_coeffs[i, 1] \
                               * self.p2**(moment_exponents[i, 1]) \
                               + moment_coeffs[i, 2] \
                               * self.p3**(moment_exponents[i, 2])

    except BaseException:
        moment_variable =   moment_coeffs[0] * self.p1**(moment_exponents[0]) \
                          + moment_coeffs[1] * self.p2**(moment_exponents[1]) \
                          + moment_coeffs[2] * self.p3**(moment_exponents[2])

    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    multiply = lambda a, b:a*b

    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    moment   =   af.sum(af.broadcast(multiply, self.f, moment_variable), 0) \
               * self.dp3 * self.dp2 * self.dp1

    af.eval(moment)
    return (moment)
