#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

def compute_moments(self, moment_name, N_s = 0, f=None, f_hat=None):
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

    N_s: The species for which you want to compute the moment quantity.
    
    f/f_hat: np.ndarray
             Pass this argument as well when you want to compute the 
             moments of the input array and not the one stored by the state vector
             of the object.

    Examples
    --------
    
    >> solver.compute_moments('density')

    Will return the density of the system at its current state.
    """
    if(self.single_mode_evolution == True):
        
        if(f is None):
            delta_moment_hat =   np.sum(getattr(self.physical_system.moment_defs, 
                                                moment_name
                                               )(self.f_hat[N_s], 
                                                 self.p1, self.p2, self.p3
                                                )
                                       )
            
            return(delta_moment_hat)
        
        else:
            
            delta_moment_hat =   np.sum(getattr(self.physical_system.moment_defs, 
                                                moment_name
                                               )(f[N_s], 
                                                 self.p1, self.p2, self.p3
                                                )
                                       )
            
            return(delta_moment_hat)

    # When evolving for several modes:
    else:
        # af.broadcast(function, *args) performs batched operations on
        # function(*args):
        if(f_hat is None and f is None):
            moment_hat = af.sum(af.broadcast(getattr(self.physical_system.moment_defs, 
                                                     moment_name
                                                    ), self.f_hat, 
                                             self.p1, self.p2, self.p3
                                            ),
                                2
                               ) * self.dp3 * self.dp2 * self.dp1

            # Scaling Appropriately:
            moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
            moment     = af.real(af.ifft2(moment_hat))
        
        elif(f_hat is not None and f is None):
            moment_hat = af.sum(af.broadcast(getattr(self.physical_system.moment_defs, 
                                                     moment_name
                                                    ), f_hat,
                                             self.p1, self.p2, self.p3
                                            ),
                                2
                               ) * self.dp3 * self.dp2 * self.dp1

            # Scaling Appropriately:
            moment_hat = 0.5 * self.N_q2 * self.N_q1 * moment_hat
            moment     = af.real(af.ifft2(moment_hat))

        elif(f_hat is None and f is not None):
            moment = af.sum(af.broadcast(getattr(self.physical_system.moment_defs, 
                                                 moment_name
                                                ), f,
                                         self.p1, self.p2, self.p3
                                        ),
                            2
                           ) * self.dp3 * self.dp2 * self.dp1

        else:
            raise BaseException('Invalid Option: Both f and f_hat cannot \
                                 be provided as arguments'
                               )

        af.eval(moment)
        return(moment)
