#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This checks that the explicit time-stepping of the
FDTD algorithm works as intended. Since Maxwell's
equation have wave like solutions, in this test we evolve
the initial state for a single timeperiod and compare the
final solution state with the initial state.

We check the fall off in error with the increase in resolution
(convergence rate) to validate the explicit FDTD algorithm.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.EM_fields_solver.fdtd_explicit import fdtd
from bolt.lib.nonlinear_solver.communicate import communicate_fields


def gauss1D(x, spread):
    return af.exp(-((x - 0.5)**2) / (2 * spread**2))


class test(object):
    def __init__(self, N):
        self.q1_start = 0
        self.q2_start = 0

        self.q1_end = 1
        self.q2_end = 1

        self.N_q1 = N
        self.N_q2 = N

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = np.random.randint(3, 5)

        self.q1 = self.q1_start \
            + (0.5 + np.arange(-self.N_ghost,self.N_q1 + self.N_ghost)) * self.dq1
        
        self.q2 = self.q2_start \
            + (0.5 + np.arange(-self.N_ghost,self.N_q2 + self.N_ghost)) * self.dq2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
        self.q2, self.q1 = af.to_array(self.q2), af.to_array(self.q1)

        self.E1_fdtd = af.constant(0, self.q1.shape[0], self.q1.shape[1],
                                   dtype=af.Dtype.f64
                                  )

        self.E2_fdtd = self.E1_fdtd.copy()
        self.E3_fdtd = self.E1_fdtd.copy()

        self.B1_fdtd = self.E1_fdtd.copy()
        self.B2_fdtd = self.E1_fdtd.copy()
        self.B3_fdtd = self.E1_fdtd.copy()

        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                               dof=6,
                                               stencil_width=self.N_ghost,
                                               boundary_type=('periodic',
                                                              'periodic'),
                                               stencil_type=1, 
                                             )

        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        self._glob_value_fields  = self._da_fields.getVecArray(self._glob_fields)
        self._local_value_fields = self._da_fields.getVecArray(self._local_fields)

        self.performance_test_flag = False

    _communicate_fields = communicate_fields


def test_fdtd_mode1():

    error_B1 = np.zeros(5)
    error_B2 = np.zeros(5)
    error_E3 = np.zeros(5)

    N = 2**np.arange(5, 10)

    for i in range(N.size):

        obj = test(N[i])

        N_g = obj.N_ghost

        obj.B1_fdtd[N_g:-N_g, N_g:-N_g] =\
            gauss1D(obj.q2[N_g:-N_g, N_g:-N_g], 0.1)

        obj.B2_fdtd[N_g:-N_g, N_g:-N_g] =\
            gauss1D(obj.q1[N_g:-N_g, N_g:-N_g], 0.1)

        dt   = obj.dq1 / 2
        time = np.arange(dt, 1 + dt, dt)

        E3_initial = obj.E3_fdtd.copy()
        B1_initial = obj.B1_fdtd.copy()
        B2_initial = obj.B2_fdtd.copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0

        for time_index, t0 in enumerate(time):
            fdtd(obj, dt)

        error_B1[i] = af.sum(af.abs(obj.B1_fdtd[N_g:-N_g, N_g:-N_g] -
                                    B1_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B1_initial.elements())

        error_B2[i] = af.sum(af.abs(obj.B2_fdtd[N_g:-N_g, N_g:-N_g] -
                                    B2_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B2_initial.elements())

        error_E3[i] = af.sum(af.abs(obj.E3_fdtd[N_g:-N_g, N_g:-N_g] -
                                    E3_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E3_initial.elements())

    poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
    poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
    poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

    assert (abs(poly_B1[0] + 3) < 0.4)
    assert (abs(poly_B2[0] + 3) < 0.4) 
    assert (abs(poly_E3[0] + 2) < 0.4)


def test_fdtd_mode2():

    error_E1 = np.zeros(3)
    error_E2 = np.zeros(3)
    error_B3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        obj = test(N[i])
        N_g = obj.N_ghost

        obj.E1_fdtd[N_g:-N_g, N_g:-N_g] =\
            gauss1D(obj.q2[N_g:-N_g, N_g:-N_g], 0.1)

        obj.E2_fdtd[N_g:-N_g, N_g:-N_g] =\
            gauss1D(obj.q1[N_g:-N_g, N_g:-N_g], 0.1)

        dt   = obj.dq1 / 2
        time = np.arange(dt, 1 + dt, dt)

        B3_initial = obj.B3_fdtd.copy()
        E1_initial = obj.E1_fdtd.copy()
        E2_initial = obj.E2_fdtd.copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0

        for time_index, t0 in enumerate(time):
            fdtd(obj, dt)

        error_E1[i] = af.sum(af.abs(obj.E1_fdtd[N_g:-N_g, N_g:-N_g] -
                                    E1_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E1_initial.elements())

        error_E2[i] = af.sum(af.abs(obj.E2_fdtd[N_g:-N_g, N_g:-N_g] -
                                    E2_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E2_initial.elements())

        error_B3[i] = af.sum(af.abs(obj.B3_fdtd[N_g:-N_g, N_g:-N_g] -
                                    B3_initial[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B3_initial.elements())

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
    poly_B3 = np.polyfit(np.log10(N), np.log10(error_B3), 1)

    assert (abs(poly_E1[0] + 3) < 0.4)
    assert (abs(poly_E2[0] + 3) < 0.4)
    assert (abs(poly_B3[0] + 2) < 0.4)
