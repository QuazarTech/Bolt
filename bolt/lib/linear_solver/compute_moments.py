#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

def compute_moments(self, moment_name, f=None, f_hat=None):
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

    f/f_hat: np.ndarray
             Pass this argument as well when you want to compute the 
             moments of the input array and not the one stored by the state vector
             of the object.

    Examples
    --------
    
    >> solver.compute_moments('density')

    The above line will lookup the definition for 'density' under the dict
    moments_exponents, and moments_coefficients and calculate the same
    accordingly

    When evolving for a single mode, this function returns moment_hat
    """
    # Checking that the moment-name is defined by the user:
    try:
        moment_exponents = \
                np.array(
                         self.physical_system.moment_exponents[moment_name]
                        )

        moment_coeffs = \
                np.array(
                         self.physical_system.moment_coeffs[moment_name]
                        )

    except BaseException:
        raise KeyError('moment_name not defined under physical system')

    # This checks that the moment definition is of the form
    # [[a, b, c], [d, e, f]...]. Alternatively if it isn't it checks
    # whether the definition is of the form [a, b, c]
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

    if(self.single_mode_evolution == True):
        if(f is None):
            delta_moment_hat =   np.sum(self.Y[0] * moment_variable) \
                               * self.dp3 * self.dp2 * self.dp1
            return(delta_moment_hat)
        
        else:
            
            moment_hat =   np.sum(f * moment_variable) \
                         * self.dp3 * self.dp2 * self.dp1
            
            return(moment_hat)

    # When evolving for several modes:
    else:
        # Since f_hat = Y[:, :, :, 0]:
        # We sum along axis 2 which contains the variations in velocity:
        # Defining lambda functions to perform broadcasting operations:
        # This is done using af.broadcast, which allows us to perform 
        # batched operations when operating on arrays of different sizes:
        multiply   = lambda a, b:a * b
        
        # af.broadcast(function, *args) performs batched operations on
        # function(*args):
        if(f_hat is None and f is None):
            moment_hat = af.sum(af.broadcast(multiply, self.Y[:, :, :, 0], 
                                             moment_variable
                                            ), 
                                2
                               ) * self.dp3 * self.dp2 * self.dp1

            # Scaling Appropriately:
            moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
            moment     = af.real(af.ifft2(moment_hat))
        
        elif(f_hat is not None and f is None):
            moment_hat = af.sum(af.broadcast(multiply, f_hat, 
                                             moment_variable
                                            ), 
                                2
                               ) * self.dp3 * self.dp2 * self.dp1

            # Scaling Appropriately:
            moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
            moment     = af.real(af.ifft2(moment_hat))

        elif(f_hat is None and f is not None):
                moment = af.sum(af.broadcast(multiply, f, 
                                             moment_variable
                                            ), 2 
                               ) * self.dp3 * self.dp2 * self.dp1
        else:
            raise BaseException('Invalid Option: Both f and f_hat cannot \
                                 be provided as arguments'
                               )

        af.eval(moment)
        return(moment)
