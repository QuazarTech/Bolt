#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the function that returns the derivative
of f w.r.t p1, p2, p3 is 4th order accurate.

This is done by checking the numerical solution against the
expected analytical solution
"""

# Importing dependencies:
import numpy as np
import arrayfire as af

# Importing solver functions:
from lib.linear_solver.calculate_dfdp_background \
    import calculate_dfdp_background


class test():
    def __init__(self, N):
        self.p_dim = 3

        self.p1_start = -10
        self.p2_start = -10
        self.p3_start = -10

        self.p1_end = 10
        self.p2_end = 10
        self.p3_end = 10

        self.N_p1 = N
        self.N_p2 = N
        self.N_p3 = N

        self.dp1 = (self.p1_end - self.p1_start) / self.N_p1
        self.dp2 = (self.p2_end - self.p2_start) / self.N_p2
        self.dp3 = (self.p3_end - self.p3_start) / self.N_p3

        self.p1 = self.p1_start + (0.5 + np.arange(self.N_p1)) * self.dp1
        self.p2 = self.p2_start + (0.5 + np.arange(self.N_p2)) * self.dp2
        self.p3 = self.p3_start + (0.5 + np.arange(self.N_p3)) * self.dp3

        self.p2, self.p1, self.p3 = np.meshgrid(self.p2, self.p1, self.p3)

        self.p1, self.p2, self.p3 = af.to_array(self.p1),\
                                    af.to_array(self.p2),\
                                    af.to_array(self.p3)

        self.p1 = af.reorder(af.flat(self.p1), 2, 3, 0, 1)
        self.p2 = af.reorder(af.flat(self.p2), 2, 3, 0, 1)
        self.p3 = af.reorder(af.flat(self.p3), 2, 3, 0, 1)


        self.f_background =   af.exp(-self.p1**2) \
                            * af.exp(-self.p2**2) \
                            * af.exp(-self.p3**2)


def test_df_dp_background():

    N = 32 * np.arange(1, 10)

    error_1 = np.zeros(N.size)
    error_2 = np.zeros(N.size)
    error_3 = np.zeros(N.size)

    for i in range(N.size):
        af.device_gc()
        obj = test(N[i])

        calculate_dfdp_background(obj)

        dfdp1_expected = -2 * obj.p1 \
                            * af.exp(-obj.p1**2) \
                            * af.exp(-obj.p2**2) \
                            * af.exp(-obj.p3**2)

        dfdp2_expected = -2 * obj.p2 \
                            * af.exp(-obj.p1**2) \
                            * af.exp(-obj.p2**2) \
                            * af.exp(-obj.p3**2)

        dfdp3_expected = -2 * obj.p3 \
                            * af.exp(-obj.p1**2) \
                            * af.exp(-obj.p2**2) \
                            * af.exp(-obj.p3**2)

        af.eval(obj.dfdp1_background, obj.dfdp2_background, obj.dfdp3_background)
        
        error_1[i] =   af.sum(af.abs(dfdp1_expected - obj.dfdp1_background)) \
                     / dfdp1_expected.elements()
        error_2[i] =   af.sum(af.abs(dfdp2_expected - obj.dfdp2_background)) \
                     / dfdp2_expected.elements()
        error_3[i] =   af.sum(af.abs(dfdp3_expected - obj.dfdp3_background)) \
                     / dfdp3_expected.elements()

                  
    poly_1 = np.polyfit(np.log10(N), np.log10(error_1), 1)
    poly_2 = np.polyfit(np.log10(N), np.log10(error_2), 1)
    poly_3 = np.polyfit(np.log10(N), np.log10(error_3), 1)

    assert(abs(poly_1[0] + 4) < 0.2)
    assert(abs(poly_2[0] + 4) < 0.2)
    assert(abs(poly_3[0] + 4) < 0.2)
