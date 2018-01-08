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

# Importing solver functions:
from bolt.lib.nonlinear_solver.compute_moments import compute_moments

moment_exponents = dict(density     = [0, 0, 0],
                        mom_p1_bulk = [1, 0, 0],
                        mom_p2_bulk = [0, 1, 0],
                        mom_p3_bulk = [0, 0, 1],
                        energy      = [2, 2, 2]
                        )

moment_coeffs = dict(density     = [1, 0, 0],
                     mom_p1_bulk = [1, 0, 0],
                     mom_p2_bulk = [0, 1, 0],
                     mom_p3_bulk = [0, 0, 1],
                     energy      = [1, 1, 1]
                     )

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
                                    {'moment_exponents': moment_exponents,
                                     'moment_coeffs': moment_coeffs
                                    }
                                   )
        self.p1_start = -10
        self.p2_start = -10
        self.p3_start = -10

        self.N_p1 = 32
        self.N_p2 = 32
        self.N_p3 = 32

        self.dp1 = (-2 * self.p1_start) / self.N_p1
        self.dp2 = (-2 * self.p2_start) / self.N_p2
        self.dp3 = (-2 * self.p3_start) / self.N_p3

        self.N_q1 = 16
        self.N_q2 = 16

        self.N_ghost = 3

        self.p1 = self.p1_start + (0.5 + np.arange(self.N_p1)) * self.dp1
        self.p2 = self.p2_start + (0.5 + np.arange(self.N_p2)) * self.dp2
        self.p3 = self.p3_start + (0.5 + np.arange(self.N_p3)) * self.dp3

        self.p2, self.p1, self.p3 = np.meshgrid(self.p2, self.p1, self.p3)

        self.p1 = af.flat(af.to_array(self.p1))
        self.p2 = af.flat(af.to_array(self.p2))
        self.p3 = af.flat(af.to_array(self.p3))

        self.q1 = (  -self.N_ghost + 0.5
                   + np.arange(self.N_q1 + 2 * self.N_ghost)
                  ) / self.N_q1

        self.q2 = (-self.N_ghost + 0.5
                   + np.arange(self.N_q2 + 2 * self.N_ghost)
                  ) / self.N_q2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)

        self.q1 = af.reorder(af.to_array(self.q1), 2, 0, 1)
        self.q2 = af.reorder(af.to_array(self.q2), 2, 0, 1)

        rho = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))
        T   = (1 + 0.01 * af.cos(2 * np.pi * self.q1 + 4 * np.pi * self.q2))

        p1_b = 0.01 * af.exp(-10 * self.q1**2 - 10 * self.q2**2)
        p2_b = 0.01 * af.exp(-10 * self.q1**2 - 10 * self.q2**2)
        p3_b = 0.01 * af.exp(-10 * self.q1**2 - 10 * self.q2**2)

        self.f = maxwell_boltzmann(rho, T, p1_b, p2_b, p3_b,
                                   self.p1, self.p2, self.p3
                                  )


def test_compute_moments():

    obj = test()
    
    rho_num = compute_moments(obj, 'density')
    rho_ana  = 1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)

    error_rho = af.mean(af.abs(rho_num - rho_ana))

    E_num = compute_moments(obj, 'energy')
    E_ana =   3 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
                * (1 + 0.01 * af.cos(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
            + 3 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
                *  (0.01 * af.exp(-10 * obj.q1**2 - 10 * obj.q2**2))**2

    error_E = af.mean(af.abs(E_num - E_ana))
    
    mom_p1b_num = compute_moments(obj, 'mom_p1_bulk')
    mom_p1b_ana =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
                  * (0.01 * af.exp(-10 * obj.q1**2 - 10 * obj.q2**2))

    error_p1b = af.mean(af.abs(mom_p1b_num - mom_p1b_ana))

    mom_p2b_num = compute_moments(obj, 'mom_p2_bulk')
    mom_p2b_ana =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
                  * (0.01 * af.exp(-10 * obj.q1**2 - 10 * obj.q2**2))

    error_p2b = af.mean(af.abs(mom_p2b_num - mom_p2b_ana))

    mom_p3b_num = compute_moments(obj, 'mom_p3_bulk')
    mom_p3b_ana  =   (1 + 0.01 * af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)) \
                   * (0.01 * af.exp(-10 * obj.q1**2 - 10 * obj.q2**2))

    error_p3b = af.mean(af.abs(mom_p3b_num - mom_p3b_ana))

    assert(error_rho < 1e-13)
    assert(error_E   < 1e-13)
    assert(error_p1b < 1e-13)
    assert(error_p2b < 1e-13)
    assert(error_p3b < 1e-13)
