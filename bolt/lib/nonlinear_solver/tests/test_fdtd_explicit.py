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
import pylab as pl
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.EM_fields_solver.fdtd_explicit import fdtd
from bolt.lib.nonlinear_solver.communicate import communicate_fields

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

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

        self.N_ghost = 3 #np.random.randint(3, 5)

        self.q1_left = self.q1_start \
            + (np.arange(-self.N_ghost,self.N_q1 + self.N_ghost)) * self.dq1
        
        self.q2_bot = self.q2_start \
            + (np.arange(-self.N_ghost,self.N_q2 + self.N_ghost)) * self.dq2

        self.q2_left_bot, self.q1_left_bot = np.meshgrid(self.q2_bot, self.q1_left)
        self.q2_left_bot, self.q1_left_bot = af.to_array(self.q2_left_bot), af.to_array(self.q1_left_bot)

        self.q1_left_bot = af.reorder(self.q1_left_bot, 2, 0, 1)
        self.q2_left_bot = af.reorder(self.q2_left_bot, 2, 0, 1)

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


def test_fdtd_mode1():

    error_B1 = np.zeros(3)
    error_B2 = np.zeros(3)
    error_E3 = np.zeros(3)

    N = np.array([32]) #2**np.arange(5, 8)

    for i in range(N.size):

        obj = test(N[i])

        N_g = obj.N_ghost

        Az = af.sin(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)

        B1_fdtd = (af.shift(Az, 0, 0, -1) - Az) / obj.dq2
        B2_fdtd = -(af.shift(Az, 0, -1) - Az) / obj.dq1

        E3_fdtd = -6 * np.pi * np.sqrt(5/9) * af.cos(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)

        obj.yee_grid_EM_fields[2] = E3_fdtd
        obj.yee_grid_EM_fields[3] = B1_fdtd
        obj.yee_grid_EM_fields[4] = B2_fdtd

        dt   = obj.dq1 * np.sqrt(9/5) / 2
        time = np.arange(dt, np.sqrt(9/5) + dt, dt)

        obj.data = np.zeros(time.size)

        E3_initial = obj.yee_grid_EM_fields[2].copy()
        B1_initial = obj.yee_grid_EM_fields[3].copy()
        B2_initial = obj.yee_grid_EM_fields[4].copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0
    
        # pl.contourf(np.array(af.reorder(obj.yee_grid_EM_fields[2, N_g:-N_g, N_g:-N_g], 1, 2, 0)), 100)
        # pl.colorbar()
        # pl.savefig('images/0000.png')
        # pl.clf()

        for time_index, t0 in enumerate(time):
            obj.time_index = time_index
            fdtd(obj, dt)

            # pl.contourf(np.array(af.reorder(obj.yee_grid_EM_fields[2, N_g:-N_g, N_g:-N_g], 1, 2, 0)), 100)
            # pl.colorbar()
            # pl.title('Time = %.2f'%t0)
            # pl.savefig('images/%04d'%(time_index+1)+'.png')
            # pl.clf()
        
        # pl.plot(time * np.sqrt(5/9), obj.data)
        # pl.xlabel('Number of Time Periods')
        # pl.ylabel(r'MEAN($|\nabla \cdot B|$)')
        # pl.savefig('plot.png')
        
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

    assert (abs(poly_B1[0] + 2) < 0.2)
    assert (abs(poly_B2[0] + 2) < 0.2) 
    assert (abs(poly_E3[0] + 2) < 0.2)


def test_fdtd_mode2():

    error_E1 = np.zeros(3)
    error_E2 = np.zeros(3)
    error_B3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        obj = test(N[i])
        N_g = obj.N_ghost

        E1_fdtd = -(2/3) * af.sin(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)
        E2_fdtd = +(1/3) * af.sin(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)

        B3_fdtd = af.sin(2 * np.pi * obj.q1_left_bot + 4 * np.pi * obj.q2_left_bot)

        obj.yee_grid_EM_fields[0, N_g:-N_g, N_g:-N_g] = E1_fdtd[:, N_g:-N_g, N_g:-N_g]
        obj.yee_grid_EM_fields[1, N_g:-N_g, N_g:-N_g] = E2_fdtd[:, N_g:-N_g, N_g:-N_g]
        obj.yee_grid_EM_fields[5, N_g:-N_g, N_g:-N_g] = B3_fdtd[:, N_g:-N_g, N_g:-N_g]

        dt   = obj.dq1 / 2
        time = np.arange(dt, 1 + dt, dt)
        
        obj.data = np.zeros(time.size)

        B3_initial = obj.yee_grid_EM_fields[5].copy()
        E1_initial = obj.yee_grid_EM_fields[0].copy()
        E2_initial = obj.yee_grid_EM_fields[1].copy()

        obj.J1, obj.J2, obj.J3 = 0, 0, 0

        for time_index, t0 in enumerate(time):
            obj.time_index = time_index
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

    print(poly_E1)
    print(poly_E2)
    print(poly_B3)

    assert (abs(poly_E1[0] + 2) < 0.4)
    assert (abs(poly_E2[0] + 2) < 0.4)
    assert (abs(poly_B3[0] + 2) < 0.4)

# print('Testing Mode 1')
# test_fdtd_mode1()

print('Testing Mode 2')
test_fdtd_mode2()
