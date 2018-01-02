#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

def compute_moments(self, moment_name, N_s = 0, f=None):
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

    N_s: The species for which you want to compute the moment quantity.

    Examples
    --------
    
    >> solver.compute_moments('density')

    The above line will lookup the definition for 'density' and calculate the same
    accordingly
    """
    N_g_p = self.N_ghost_p

    # Number of DOF in the array for a single species:
    dof =   (self.N_p1 + 2 * N_g_p) \
          * (self.N_p2 + 2 * N_g_p) \
          * (self.N_p3 + 2 * N_g_p)

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

    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    multiply = lambda a, b:a*b

    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    if(f is None):
        try:
            N_q1 = self.f.shape[1]
            N_q2 = self.f.shape[2]

        except:
            N_q1 = N_q2 = 1

        if(N_g_p != 0):

            f = af.moddims(self.\
                           _convert_to_p_expanded(self.f[(N_s-1) * dof:N_s * dof,
                                                         :, :
                                                        ]
                                                 )[N_g_p:-N_g_p, 
                                                   N_g_p:-N_g_p,
                                                   N_g_p:-N_g_p
                                                  ],
                           self.N_p1 * self.N_p2 * self.N_p3, 
                           N_q1, N_q2
                          )

        else:
            f = self.f[N_s * dof:(N_s + 1) * dof,
                       :, :
                      ]
    
    else:

        try:
            N_q1 = f.shape[1]
            N_q2 = f.shape[2]
        except:
            N_q1 = N_q2 = 1

        if(N_g_p != 0):
            f = af.moddims(self.\
                           _convert_to_p_expanded(f[N_s * dof:(N_s + 1) * dof,
                                                    :, :
                                                   ]
                                                 )[N_g_p:-N_g_p, 
                                                   N_g_p:-N_g_p,
                                                   N_g_p:-N_g_p
                                                  ],
                           self.N_p1 * self.N_p2 * self.N_p3, 
                           N_q1, N_q2
                          )

    moment   =   af.sum(af.broadcast(getattr(self.physical_system.moment_defs, 
                                             moment_name
                                            ), f, p1, p2, p3
                                    ),
                        0
                       ) * self.dp3 * self.dp2 * self.dp1

    af.eval(moment)
    return (moment)
