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
                   user under moments under src and passed to the 
                   physical_system object.
    
    f: af.Array
       Pass this argument as well when you want to compute the 
       moments of the input array and not the one stored by the state vector
       of the object.

    Examples
    --------
    
    >> solver.compute_moments('density')

    The above line will lookup the definition for 'density' and calculate the same
    accordingly
    """
    p1 = self.p1_center
    p2 = self.p2_center
    p3 = self.p3_center

    if(f is None):
        f = self.f
        
    moment = af.broadcast(getattr(self.physical_system.moments, 
                                  moment_name
                                 ), f, p1, p2, p3, self.dp3 * self.dp2 * self.dp1
                         )

    af.eval(moment)
    return (moment)
