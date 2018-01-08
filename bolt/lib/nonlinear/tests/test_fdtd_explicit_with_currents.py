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

        self.N_ghost = 3 #np.random.randint(3, 5)

        self.q1_left = self.q1_start \
            + (np.arange(-self.N_ghost,self.N_q1 + self.N_ghost)) * self.dq1

        self.q1_center = self.q1_start \
            + (0.5 + np.arange(-self.N_ghost,self.N_q1 + self.N_ghost)) * self.dq1
        
        self.q2_bot = self.q2_start \
            + (np.arange(-self.N_ghost,self.N_q2 + self.N_ghost)) * self.dq2

        self.q2_center = self.q2_start \
            + (0.5 + np.arange(-self.N_ghost,self.N_q2 + self.N_ghost)) * self.dq2

        self.q2_left_bot, self.q1_left_bot = np.meshgrid(self.q2_bot, self.q1_left)
        self.q2_left_center, self.q1_left_center = np.meshgrid(self.q2_center, self.q1_left)
        self.q2_center_bot, self.q1_center_bot = np.meshgrid(self.q2_bot, self.q1_center)

        self.q2_left_bot, self.q1_left_bot = af.to_array(self.q2_left_bot), af.to_array(self.q1_left_bot)
        self.q2_left_center, self.q1_left_center = af.to_array(self.q2_left_center), af.to_array(self.q1_left_center)
        self.q2_center_bot, self.q1_center_bot = af.to_array(self.q2_center_bot), af.to_array(self.q1_center_bot)

        self.q1_left_bot = af.reorder(self.q1_left_bot, 2, 0, 1)
        self.q2_left_bot = af.reorder(self.q2_left_bot, 2, 0, 1)

        self.q1_left_center = af.reorder(self.q1_left_center, 2, 0, 1)
        self.q2_left_center = af.reorder(self.q2_left_center, 2, 0, 1)

        self.q1_center_bot = af.reorder(self.q1_center_bot, 2, 0, 1)
        self.q2_center_bot = af.reorder(self.q2_center_bot, 2, 0, 1)

        self.yee_grid_EM_fields = af.constant(0, 6, self.q1_left_bot.shape[1], 
                                              self.q2_left_bot.shape[2],
                                              dtype=af.Dtype.f64
                                             )

        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                               dof=6,
                                               stencil_width=self.N_ghost,
                                               boundary_type=('periodic',
                                                              'periodic'),
                                               stencil_type=1, 
                                             )

        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        self._glob_fields_array  = self._glob_fields.getArray()
        self._local_fields_array = self._local_fields.getArray()

        self.boundary_conditions = type('obj', (object, ),
                                        {'in_q1':'periodic',
                                         'in_q2':'periodic'
                                        }
                                       )

        self.performance_test_flag = False

    _communicate_fields = communicate_fields


def test_fdtd_modeTM():

    error_B1 = np.zeros(3)
    error_B2 = np.zeros(3)
    error_E3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        obj = test(N[i])

        dt   = obj.dq1 / 2
        time = np.arange(dt, 1 + dt, dt)

        N_g = obj.N_ghost

        A3 = af.sin(2 * np.pi * (obj.q1_left_bot - 0.5 * dt) + 4 * np.pi * (obj.q2_left_bot - 0.5 * dt))

        B1_fdtd = (af.shift(A3, 0, 0, -1) - A3) / obj.dq2
        B2_fdtd = -(af.shift(A3, 0, -1) - A3) / obj.dq1

        E3_fdtd = 6 * np.pi * af.cos(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)

        obj.yee_grid_EM_fields[2] = E3_fdtd
        obj.yee_grid_EM_fields[3] = B1_fdtd
        obj.yee_grid_EM_fields[4] = B2_fdtd


        E3_initial = obj.yee_grid_EM_fields[2].copy()
        B1_initial = obj.yee_grid_EM_fields[3].copy()
        B2_initial = obj.yee_grid_EM_fields[4].copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0
    
        for time_index, t0 in enumerate(time):
            obj.J3 = -16 * np.pi**2 * af.sin(  2 * np.pi * (obj.q1_left_bot - t0 + 0.5 * dt) 
                                             + 4 * np.pi * (obj.q2_left_bot - t0 + 0.5 * dt)
                                            )
            fdtd(obj, dt)
        
        error_B1[i] = af.sum(af.abs(obj.yee_grid_EM_fields[3, N_g:-N_g, N_g:-N_g] -
                                    B1_initial[0, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B1_initial.elements())

        error_B2[i] = af.sum(af.abs(obj.yee_grid_EM_fields[4, N_g:-N_g, N_g:-N_g] -
                                    B2_initial[0, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B2_initial.elements())

        error_E3[i] = af.sum(af.abs(obj.yee_grid_EM_fields[2, N_g:-N_g, N_g:-N_g] -
                                    E3_initial[0, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E3_initial.elements())

    poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
    poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
    poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

    assert (abs(poly_B1[0] + 2) < 0.25)
    assert (abs(poly_B2[0] + 2) < 0.25) 
    assert (abs(poly_E3[0] + 2) < 0.25)


def test_fdtd_modeTE():

    error_E1 = np.zeros(3)
    error_E2 = np.zeros(3)
    error_B3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        obj = test(N[i])

        dt   = obj.dq1 / 2
        time = np.arange(dt, 1 + dt, dt)

        N_g = obj.N_ghost

        E1_fdtd = 8 * np.pi * af.cos(4 * np.pi * obj.q2_center_bot)
        E2_fdtd = 4 * np.pi * af.cos(2 * np.pi * obj.q1_left_center)

        B3_fdtd =   2 * np.pi * af.cos(2 * np.pi * (obj.q1_left_center - dt)) \
                  - 4 * np.pi * af.cos(4 * np.pi * (obj.q2_center_bot - dt))

        obj.yee_grid_EM_fields[0, N_g:-N_g, N_g:-N_g] = E1_fdtd[:, N_g:-N_g, N_g:-N_g]
        obj.yee_grid_EM_fields[1, N_g:-N_g, N_g:-N_g] = E2_fdtd[:, N_g:-N_g, N_g:-N_g]
        obj.yee_grid_EM_fields[5, N_g:-N_g, N_g:-N_g] = B3_fdtd[:, N_g:-N_g, N_g:-N_g]

        
        B3_initial = obj.yee_grid_EM_fields[5].copy()
        E1_initial = obj.yee_grid_EM_fields[0].copy()
        E2_initial = obj.yee_grid_EM_fields[1].copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0

        for time_index, t0 in enumerate(time):

            obj.J1 = 48 * np.pi**2 * af.sin(4 * np.pi * (obj.q2_center_bot - 2 * t0))
            obj.J2 = 12 * np.pi**2 * af.sin(2 * np.pi * (obj.q1_left_center - 2 * t0))
            
            fdtd(obj, dt)

        error_E1[i] = af.sum(af.abs(obj.yee_grid_EM_fields[0, N_g:-N_g, N_g:-N_g] -
                                    E1_initial[:, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E1_initial.elements())

        error_E2[i] = af.sum(af.abs(obj.yee_grid_EM_fields[1, N_g:-N_g, N_g:-N_g] -
                                    E2_initial[:, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E2_initial.elements())

        error_B3[i] = af.sum(af.abs(obj.yee_grid_EM_fields[5, N_g:-N_g, N_g:-N_g] -
                                    B3_initial[:, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B3_initial.elements())

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
    poly_B3 = np.polyfit(np.log10(N), np.log10(error_B3), 1)

    assert (abs(poly_E1[0] + 2) < 0.25)
    assert (abs(poly_E2[0] + 2) < 0.25)
    assert (abs(poly_B3[0] + 2) < 0.25)
