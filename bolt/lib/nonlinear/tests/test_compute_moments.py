#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test we check that the compute_moments returns
the correct values of the moments for well defined moment
exponents and coeffs as we expect.

For this check we take the density, temperature, and all bulk
velocities to some analytical function of q1 and q2 which is 
then checked against the values returned by the compute_moments
function
"""

# Importing dependencies:
import numpy as np
import arrayfire as af

from bolt.lib.nonlinear.compute_moments import compute_moments
from bolt.lib.utils.calculate_q import calculate_q_center
from bolt.lib.utils.calculate_p import calculate_p_center

import input_files.moments as moments

# Wrapping the function using the broadcasting function which allows
# batched operations on arrays of different sizes:
@af.broadcast
def maxwell_boltzmann(rho, T, p1_b, p2_b, p3_b, p1, p2, p3):

    f = rho * (1 / (2 * np.pi * T))**(3 / 2) \
            * af.exp(-1 * (p1 - p1_b)**2 / (2 * T)) \
            * af.exp(-1 * (p2 - p2_b)**2 / (2 * T)) \
            * af.exp(-1 * (p3 - p3_b)**2 / (2 * T))

    af.eval(f)
    return(f)

class test(object):
    def __init__(self):
        self.physical_system = type('obj', (object,),
                                    {'moments': moments}
                                   )
        self.p1_start = [-10]
        self.p2_start = [-10]
        self.p3_start = [-10]

        self.N_p1 = 32
        self.N_p2 = 32
        self.N_p3 = 32

        self.dp1 = (-2 * self.p1_start[0]) / self.N_p1
        self.dp2 = (-2 * self.p2_start[0]) / self.N_p2
        self.dp3 = (-2 * self.p3_start[0]) / self.N_p3

        self.q1_start = 0; self.q2_start = 0
        self.q1_end   = 1; self.q2_end   = 1

        self.N_q1 = 16
        self.N_q2 = 16

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.q1_center, self.q2_center = \
            calculate_q_center(self.q1_start, self.q2_start,
                               self.N_q1, self.N_q2, 0,
                               self.dq1, self.dq2
                              )

        self.p1_center, self.p2_center, self.p3_center = \
            calculate_p_center(self.p1_start, self.p2_start, self.p3_start,
                               self.N_p1, self.N_p2, self.N_p3,
                               [self.dp1], [self.dp2], [self.dp3]
                              )

        rho = (1 + 0.01 * af.sin(  2 * np.pi * self.q1_center 
                                 + 4 * np.pi * self.q2_center
                                )
              )
        
        T   = (1 + 0.01 * af.cos(  2 * np.pi * self.q1_center 
                                 + 4 * np.pi * self.q2_center
                                )
              )

        p1_b = 0.01 * af.exp(-10 * self.q1_center**2 - 10 * self.q2_center**2)
        p2_b = 0.01 * af.exp(-10 * self.q1_center**2 - 10 * self.q2_center**2)
        p3_b = 0.01 * af.exp(-10 * self.q1_center**2 - 10 * self.q2_center**2)

        self.f = maxwell_boltzmann(rho, T, p1_b, p2_b, p3_b,
                                   self.p1_center, 
                                   self.p2_center, 
                                   self.p3_center
                                  )


def test_compute_moments():

    obj = test()
    
    rho_num = compute_moments(obj, 'density')
    rho_ana  = 1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)

    error_rho = af.mean(af.abs(rho_num - rho_ana))

    E_num = compute_moments(obj, 'energy')
    E_ana =   3/2 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
                  * (1 + 0.01 * af.cos(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
            + 3/2 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
                  * (0.01 * af.exp(-10 * obj.q1_center**2 - 10 * obj.q2_center**2))**2

    error_E = af.mean(af.abs(E_num - E_ana))
    
    mom_p1b_num = compute_moments(obj, 'mom_v1_bulk')
    mom_p1b_ana =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
                  * (0.01 * af.exp(-10 * obj.q1_center**2 - 10 * obj.q2_center**2))

    error_p1b = af.mean(af.abs(mom_p1b_num - mom_p1b_ana))

    mom_p2b_num = compute_moments(obj, 'mom_v2_bulk')
    mom_p2b_ana =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
                  * (0.01 * af.exp(-10 * obj.q1_center**2 - 10 * obj.q2_center**2))

    error_p2b = af.mean(af.abs(mom_p2b_num - mom_p2b_ana))

    mom_p3b_num = compute_moments(obj, 'mom_v3_bulk')
    mom_p3b_ana  =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)) \
                   * (0.01 * af.exp(-10 * obj.q1_center**2 - 10 * obj.q2_center**2))

    error_p3b = af.mean(af.abs(mom_p3b_num - mom_p3b_ana))

    print(error_rho)
    print(error_E)
    print(error_p1b)
    print(error_p2b)
    print(error_p3b)

    print((error_rho + error_E + error_p1b + error_p2b + error_p3b) / 5)

    assert(error_rho < 1e-13)
    assert(error_E   < 1e-13)
    assert(error_p1b < 1e-13)
    assert(error_p2b < 1e-13)
    assert(error_p3b < 1e-13)

test_compute_moments()
