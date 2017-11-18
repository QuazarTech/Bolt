"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = af.select(af.abs(q2)>0.25, q1**0, 2)

    p1_bulk  =   af.select(af.abs(q2)>0.25, -0.5 * q1**0, 0.5) \
   
    
    # Seeding the instability
    p1_bulk = af.reorder(p1_bulk, 1, 2, 0, 3)
    p1_bulk +=  af.to_array(  np.random.rand(1, q1.shape[1], q1.shape[2])
                            * np.random.choice([-1, 1], size = (1, q1.shape[1], q1.shape[2]))
                           ) * 0.005
              
    p2_bulk = af.to_array(  np.random.rand(1, q1.shape[1], q1.shape[2])
                          * np.random.choice([-1, 1], size = (1, q1.shape[1], q1.shape[2]))
                         ) * 0.005
    
    p1_bulk = af.reorder(p1_bulk, 2, 0, 1, 3)
    p2_bulk = af.reorder(p2_bulk, 2, 0, 1, 3)

    T = (2.5 / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p3)**2 / (2 * k * T))

    f  = af.moddims(f, 32**2, 16, 16, 16)
    p1 = af.moddims(f, 32**2, 16, 16, 16)

    import pylab as pl 
    pl.plot(np.swapaxes(np.array(f)[0, :, 0, 0], 0, 1), np.swapaxes(np.array(f)[0, :, 0, 0], 0, 1))
    pl.savefig(plot.png)
    exit()

    af.eval(f)
    return (f)
