#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
from bolt.lib.utils.fft_funcs import fft2, ifft2
import numpy as np

# TODO: Change docstring to say that it returns either moment_hat or moment
# depending on the input
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
                   user under moments under src and passed to the 
                   physical_system object.

    f/f_hat: np.ndarray
             Pass this argument as well when you want to compute the 
             moments of the input array and not the one stored by the state vector
             of the object.

    Examples
    --------
    
    >> solver.compute_moments('density')

    Will return the density of the system at its current state.
    """
    if(f_hat is None and f is None):
        # af.broadcast(function, *args) performs batched operations on function(*args):
        moment_hat = af.broadcast(getattr(self.physical_system.moments, 
                                          moment_name
                                         ), self.f_hat, 
                                  self.p1_center, self.p2_center, self.p3_center,
                                  self.dp3 * self.dp2 * self.dp1
                                 )

        # Scaling Appropriately:
        moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
        moment     = af.real(ifft2(moment_hat))
        
        af.eval(moment)
        return(moment)
    
    elif(f_hat is not None and f is None):
        moment_hat = af.broadcast(getattr(self.physical_system.moments, 
                                          moment_name
                                         ), f_hat,
                                  self.p1_center, self.p2_center, self.p3_center,
                                  self.dp3 * self.dp2 * self.dp1
                                 )

        af.eval(moment_hat)
        return(moment_hat)

    elif(f_hat is None and f is not None):
        moment = af.broadcast(getattr(self.physical_system.moments, 
                                      moment_name
                                     ), f,
                              self.p1_center, self.p2_center, self.p3_center,
                              self.dp3 * self.dp2 * self.dp1
                             )
        af.eval(moment)
        return(moment)

    else:
        raise BaseException('Invalid Option: Both f and f_hat cannot \
                             be provided as arguments'
                           )
