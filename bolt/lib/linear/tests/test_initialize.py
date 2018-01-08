#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test, we ensure that the initialize function works
as we would expect. We check that the f_background is obtained
correctly when a perturbed distribution function is given as
input
"""

# Importing dependencies:
import numpy as np
from numpy.fft import fftfreq
import arrayfire as af

# Importing solver functions:
from bolt.lib.linear_solver.linear_solver import linear_solver
from bolt.lib.linear_solver.compute_moments \
    import compute_moments as compute_moments_imported

initialize = linear_solver._initialize

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

def empty_function(*args):
    return None

def MB_dist(q1, q2, p1, p2, p3, params):

    # Calculating the perturbed density:
    rho = 1 + (0.01 * af.cos(2 * np.pi * q1 + 4 * np.pi * q2))

    f = rho * (1 / (2 * np.pi))**(3 / 2) \
            * af.exp(-0.5 * p1**2) \
            * af.exp(-0.5 * p2**2) \
            * af.exp(-0.5 * p3**2)

    af.eval(f)
    return (f)

class params:
    def __init__(self):
        pass

class test(object):
    def __init__(self):
        # Initializing an object with required attributes:
        self.physical_system = type('obj', (object, ),
                                    {'initial_conditions':
                                      type('obj', (object,), {'initialize_f':MB_dist}),
                                     'params':
                                      type('obj', (object,), {'fields_initialize':'fft',
                                                              'charge_electron':  -1
                                                             }
                                          ),
                                      'moment_exponents': moment_exponents,
                                      'moment_coeffs':    moment_coeffs
                                     }
                                    )

        self.single_mode_evolution = False

        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = 33
        self.N_q2 = 33 
        
        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.p1_start = np.random.randint(-10, -5)
        self.p2_start = np.random.randint(-10, -5)
        self.p3_start = np.random.randint(-10, -5)

        self.p1_end = np.random.randint(5, 10)
        self.p2_end = np.random.randint(5, 10)
        self.p3_end = np.random.randint(5, 10)

        self.N_p1 = np.random.randint(16, 32)
        self.N_p2 = np.random.randint(16, 32)
        self.N_p3 = np.random.randint(16, 32)

        self.dp1 = (self.p1_end - self.p1_start) / self.N_p1
        self.dp2 = (self.p2_end - self.p2_start) / self.N_p2
        self.dp3 = (self.p3_end - self.p3_start) / self.N_p3

        self.q1_center, self.q2_center = linear_solver._calculate_q_center(self)
        self.p1, self.p2, self.p3      = linear_solver._calculate_p_center(self)
        self.k_q1, self.k_q2           = linear_solver._calculate_k(self)

    _calculate_dfdp_background = empty_function
    compute_moments            = compute_moments_imported

def test_initialize():
    obj = test()
    initialize(obj, params)
    f_background_ana =    (1 / (2 * np.pi))**(3 / 2) \
                        * af.exp(-0.5 * obj.p1**2)  \
                        * af.exp(-0.5 * obj.p2**2)  \
                        * af.exp(-0.5 * obj.p3**2)

    assert(af.sum(af.abs(obj.f_background - f_background_ana))<1e-13)
