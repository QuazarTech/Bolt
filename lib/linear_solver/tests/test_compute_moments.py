#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import numpy as np
import arrayfire as af
import pylab as pl

# Importing solver functions:
from lib.linear_solver.compute_moments import compute_moments

# In this test we check that the compute_moments returns the correct
# values of the moments for well defined moment exponents and coeffs
# as we expect.

moment_exponents = dict(density=[0, 0, 0],
                        mom_p1_bulk=[1, 0, 0],
                        mom_p2_bulk=[0, 1, 0],
                        mom_p3_bulk=[0, 0, 1],
                        energy=[2, 2, 2]
                        )

moment_coeffs = dict(density=[1, 0, 0],
                     mom_p1_bulk=[1, 0, 0],
                     mom_p2_bulk=[0, 1, 0],
                     mom_p3_bulk=[0, 0, 1],
                     energy=[1, 1, 1]
                     )


class test(object):
    def __init__(self):
        self.physical_system = type('obj', (object,),
                                    {'moment_exponents': moment_exponents,
                                     'moment_coeffs': moment_coeffs}
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

        self.p1 = af.tile(
            af.reorder(
                af.flat(
                    af.to_array(
                        self.p1)),
                2,
                3,
                0,
                1),
            self.N_q1 +
            2 *
            self.N_ghost,
            self.N_q2 +
            2 *
            self.N_ghost,
            1)

        self.p2 = af.tile(
            af.reorder(
                af.flat(
                    af.to_array(
                        self.p2)),
                2,
                3,
                0,
                1),
            self.N_q1 +
            2 *
            self.N_ghost,
            self.N_q2 +
            2 *
            self.N_ghost,
            1)

        self.p3 = af.tile(
            af.reorder(
                af.flat(
                    af.to_array(
                        self.p3)),
                2,
                3,
                0,
                1),
            self.N_q1 +
            2 *
            self.N_ghost,
            self.N_q2 +
            2 *
            self.N_ghost,
            1)

        self.q1 = (-self.N_ghost + 0.5 +
                   np.arange(self.N_q1 + 2 * self.N_ghost)) / self.N_q1
        self.q2 = (-self.N_ghost + 0.5 +
                   np.arange(self.N_q2 + 2 * self.N_ghost)) / self.N_q2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)

        self.q1 = af.tile(
            af.to_array(
                self.q1),
            1,
            1,
            self.N_p1 *
            self.N_p2 *
            self.N_p3)
        self.q2 = af.tile(
            af.to_array(
                self.q2),
            1,
            1,
            self.N_p1 *
            self.N_p2 *
            self.N_p3)

        rho = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))
        T = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))

        p1_b = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))
        p2_b = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))
        p3_b = (1 + 0.01 * af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))

        self.f = rho * (1 / (2 * np.pi * T))**(3 / 2) * \
            af.exp(-1 * (self.p1 - p1_b)**2 / (2 * T)) * \
            af.exp(-1 * (self.p2 - p2_b)**2 / (2 * T)) * \
            af.exp(-1 * (self.p3 - p3_b)**2 / (2 * T))

        self.Y = 2 * af.fft2(self.f) / (self.N_q1 * self.N_q2)


def test_compute_moments():
    obj = test()
    rho_calc = compute_moments(obj, 'density')
    error_rho = af.sum(af.abs(rho_calc - (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                            4 * np.pi * obj.q2[:, :, 0]
                                                            )
                                          ))) / rho_calc.elements()

    E_calc = compute_moments(obj, 'energy')

    error_E = af.sum(af.abs(E_calc - 3 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                            4 * np.pi * obj.q2[:, :, 0]
                                                            )
                                          )**2

                                   - 3 * (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                            4 * np.pi * obj.q2[:, :, 0]
                                                            )
                                          )**3

                            )
                     ) / rho_calc.elements()

    mom_p1_calc = compute_moments(obj, 'mom_p1_bulk')
    mom_p2_calc = compute_moments(obj, 'mom_p2_bulk')
    mom_p3_calc = compute_moments(obj, 'mom_p3_bulk')

    error_p1 = af.sum(af.abs(mom_p1_calc - (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                              4 * np.pi * obj.q2[:, :, 0]
                                                              )
                                            ))**2) / rho_calc.elements()

    error_p2 = af.sum(af.abs(mom_p2_calc - (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                              4 * np.pi * obj.q2[:, :, 0]
                                                              )
                                            ))**2) / rho_calc.elements()

    error_p3 = af.sum(af.abs(mom_p3_calc - (1 + 0.01 * af.sin(2 * np.pi * obj.q1[:, :, 0] +
                                                              4 * np.pi * obj.q2[:, :, 0]
                                                              )
                                            ))**2) / rho_calc.elements()

    assert(error_rho < 1e-13)
    assert(error_E < 1e-13)
