#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the convergence for the interpolation routine
falls off with N^{-2}, where N is the number of divisions chosen.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

from lib.nonlinear_solver.nonlinear_solver import nonlinear_solver
from lib.nonlinear_solver.interpolation_routines import f_interp_p_3d

convert_imported = nonlinear_solver._convert


class test(object):
    def __init__(self, N):
        self.N_p1 = N
        self.N_p2 = N
        self.N_p3 = N

        self.dp1 = 20 / self.N_p1
        self.dp2 = 20 / self.N_p2
        self.dp3 = 20 / self.N_p3

        self.N_ghost = N_ghost = 3

        self.p1_start = self.p2_start = self.p3_start = -10

        p1_center = self.p1_start + \
            (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
        p2_center = self.p2_start + \
            (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
        p3_center = self.p3_start + \
            (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center, p1_center,
                                                      p3_center)

        p1_center = af.flat(af.to_array(p1_center))
        p2_center = af.flat(af.to_array(p2_center))
        p3_center = af.flat(af.to_array(p3_center))

        self.p1 = af.tile(
            af.reorder(p1_center, 2, 3, 0, 1), 8 + 2 * N_ghost,
            8 + 2 * N_ghost, 1, 1)

        self.p2 = af.tile(
            af.reorder(p2_center, 2, 3, 0, 1), 8 + 2 * N_ghost,
            8 + 2 * N_ghost, 1, 1)

        self.p3 = af.tile(
            af.reorder(p3_center, 2, 3, 0, 1), 8 + 2 * N_ghost,
            8 + 2 * N_ghost, 1, 1)

        self.q1_start = self.q2_start = 0

        q1_center = af.to_array(
            (-N_ghost + np.arange(8 + 2 * N_ghost) + 0.5) * (1 / 8))
        q2_center = af.to_array(
            (-N_ghost + np.arange(8 + 2 * N_ghost) + 0.5) * (1 / 8))

        # Tiling such that variation in q1 is along axis 0:
        q1_center = af.tile(q1_center, 1, 8 + 2 * self.N_ghost,
                            self.N_p1 * self.N_p2 * self.N_p3)

        # Tiling such that variation in q2 is along axis 1:
        q2_center = af.tile(
            af.reorder(q2_center), 8 + 2 * self.N_ghost, 1,
            self.N_p1 * self.N_p2 * self.N_p3, 1)

        self.q1_center, self.q2_center = q1_center, q2_center

        # Creating Dummy Values:
        self.E1 = self.q1_center[:, :, 0, 0]
        self.E2 = self.q1_center[:, :, 0, 0]
        self.E3 = self.q1_center[:, :, 0, 0]

        self.B1 = self.q1_center[:, :, 0, 0]
        self.B2 = self.q1_center[:, :, 0, 0]
        self.B3 = self.q1_center[:, :, 0, 0]

        self.f = af.sin(2 * np.pi * self.p1 + 4 * np.pi * self.p2 +
                        6 * np.pi * self.p3)

        self.physical_system = type('obj', (object, ),
                                    {'params': 'placeHolder'})

        self._da = PETSc.DMDA().create(
            [8, 8],
            dof=(self.N_p1 * self.N_p2 * self.N_p3),
            stencil_width=N_ghost)

    def _A_p(self, *args):
        return (1, 1, 1)

    _convert = convert_imported


def test_f_interp_p_3d():
    N = np.array([16, 24, 32, 48, 64, 96, 128])
    error = np.zeros(N.size)

    for i in range(N.size):
        obj = test(int(N[i]))
        f_interp_p_3d(obj, 0.00001)
        f_analytic = af.sin(2 * np.pi * (obj.p1 - 0.00001) + 4 * \
                            np.pi * (obj.p2 - 0.00001) + 6 * np.pi * (obj.p3 - 0.00001))
        error[i] = af.sum(af.abs(obj.f[3:-3, 3:-3] - f_analytic[3:-3, 3:-3])
                          ) / f_analytic[3:-3, 3:-3].elements()

    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    print(poly)
    print(error)
    # assert(abs(poly[0] + 2)<0.2)


test_f_interp_p_3d()
