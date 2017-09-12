#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the fields solver works correctly.
The solution as yielded by the fields solver is checked against
the analytical solution, and checked that the error is around
machine precision.
"""

# Importing dependencies:
import numpy as np
import arrayfire as af
from numpy.fft import fftfreq

# Importing solver functions:
from bolt.lib.linear_solver.EM_fields_solver \
    import compute_electrostatic_fields


class test(object):
    def __init__(self):
        self.physical_system = type('obj', (object, ),
                                    {'params': type('obj', (object, ),
                                        {'charge_electron': -1})
                                     })

        self.N_q1 = 32
        self.N_q2 = 64

        self.N_p1 = 2
        self.N_p2 = 3
        self.N_p3 = 4

        self.k_q1 = 2 * np.pi * fftfreq(self.N_q1, 1 / self.N_q1)
        self.k_q2 = 2 * np.pi * fftfreq(self.N_q2, 1 / self.N_q2)

        self.k_q2, self.k_q1 = np.meshgrid(self.k_q2, self.k_q1)
        self.k_q2, self.k_q1 = af.to_array(self.k_q2), af.to_array(self.k_q1)

        self.q1 = af.to_array((0.5 + np.arange(self.N_q1)) * (1 / self.N_q1))
        self.q2 = af.to_array((0.5 + np.arange(self.N_q2)) * (1 / self.N_q2))

        self.q1 = af.tile(self.q1, 1, self.N_q2)
        self.q2 = af.tile(af.reorder(self.q2), self.N_q1, 1)

    def compute_moments(self, string):
        return(1
               + 0.01 * af.cos(  2 * np.pi * self.q1
                               + 4 * np.pi * self.q2
                              )
             
               - 0.02 * af.sin(  2 * np.pi * self.q1
                               + 4 * np.pi * self.q2
                              )
              )

def test_compute_electrostatic_fields():

    test_obj = test()
    compute_electrostatic_fields(test_obj)

    E1 = 0.5 * test_obj.N_q1 * test_obj.N_q2 * af.ifft2(test_obj.E1_hat)
    E2 = 0.5 * test_obj.N_q1 * test_obj.N_q2 * af.ifft2(test_obj.E2_hat)

    E1_analytical =   test_obj.physical_system.params.charge_electron \
                    * 2 * np.pi / (20 * np.pi**2) \
                    * (  0.01 * af.sin(2 * np.pi * test_obj.q1 + 4 * np.pi * test_obj.q2)
                       + 0.02 * af.cos(2 * np.pi * test_obj.q1 + 4 * np.pi * test_obj.q2)
                      )

    E2_analytical =   test_obj.physical_system.params.charge_electron \
                    * 4 * np.pi / (20 * np.pi**2) \
                    * (  0.01 * af.sin(2 * np.pi * test_obj.q1 + 4 * np.pi * test_obj.q2)
                       + 0.02 * af.cos(2 * np.pi * test_obj.q1 + 4 * np.pi * test_obj.q2)
                      )

    add = lambda a,b:a+b

    error_E1 = af.sum(af.abs(af.broadcast(add, E1_analytical, - E1)))/E1.elements()
    error_E2 = af.sum(af.abs(af.broadcast(add, E2_analytical, - E2)))/E2.elements()

    assert(error_E1 < 1e-14)
    assert(error_E2 < 1e-14)
