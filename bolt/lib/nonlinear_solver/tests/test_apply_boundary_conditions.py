#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test, we ensure that the apply_boundary_condition() functions
as intended. As of writing the test the following boundary conditions have been tested

Dirichet B.Cs - Here it needs to be ensured that the ghost zones have the value of the edge zone
only when the world lines go out of the domain. When the world lines are inside the computational
domain, then the distribution function values are assigned using interpolation.

Periodic B.Cs - The test over here ensures that the periodic boundary conditions are being applied
correctly:f(x + L) = f(x) where L is the periodicity of the domain.

Mirror B.Cs - The enforcement of the mirror boundary conditions is explored through the example below:
# | o | o | o || o | o | o |...
#   0   1   2    3   4   5
For mirror B.Cs:
f(i = 0) = f(i = 5)
f(i = 1) = f(i = 4)
f(i = 2) = f(i = 3)
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.apply_boundary_conditions \
    import apply_bcs_f

from bolt.lib.nonlinear_solver.communicate import communicate_f

def f_x(*args):
    return(1)

def f_y(*args):
    return(2)

class test(object):
    def __init__(self, in_q1, in_q2):
        self.physical_system = type('obj', (object, ),
                                    {'boundary_conditions': type('obj', (object, ),
                                     {'in_q1': in_q1, 'in_q2': in_q2,
                                      'f_left':f_x, 'f_right':f_x,
                                      'f_bot':f_y, 'f_top':f_y,
                                     }),
                                     'params':'placeHolder'
                                    }
                                   )

        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = np.random.randint(16, 32)
        self.N_q2 = np.random.randint(16, 32)

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = np.random.randint(1, 5)

        self.N_p1 = np.random.randint(16, 32)
        self.N_p2 = np.random.randint(16, 32)
        self.N_p3 = np.random.randint(16, 32)

        # Dummy variable:
        self.p1, self.p2, self.p3 = 1, 1, 1

        self.q1_center =   self.q1_start \
                         + (0.5 + np.arange(-self.N_ghost,
                                             self.N_q1 + self.N_ghost
                                           )
                           ) * self.dq1

        self.q2_center =   self.q2_start \
                         + (0.5 + np.arange(-self.N_ghost,
                                             self.N_q2 + self.N_ghost
                                           )
                           ) * self.dq2

        self.q1_center = af.tile(af.to_array(self.q1_center), 1,
                                 self.N_q2 + 2 * self.N_ghost,
                                 self.N_p1 * self.N_p2 * self.N_p3
                                )

        self.q2_center = af.tile(af.reorder(af.to_array(self.q2_center)),
                                 self.N_q1 + 2 * self.N_ghost, 1,
                                 self.N_p1 * self.N_p2 * self.N_p3
                                )

        petsc_in_q1 = 'periodic'
        petsc_in_q2 = 'periodic'

        if(in_q1 != 'periodic'):
            petsc_in_q1 = 'ghosted'
            petsc_in_q2 = 'ghosted'

        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof=(self.N_p1 * self.N_p2 * self.N_p3),
                                         stencil_width=self.N_ghost,
                                         boundary_type=(petsc_in_q1, petsc_in_q2),
                                         stencil_type=1, 
                                        )

        self._glob_f  = self._da_f.createGlobalVec()
        self._local_f = self._da_f.createLocalVec()

        self._glob_value_f  = self._da_f.getVecArray(self._glob_f)
        self._local_value_f = self._da_f.getVecArray(self._local_f)

        self.f = af.constant(0,
                             self.q1_center.shape[0],
                             self.q1_center.shape[1],
                             self.q1_center.shape[2],
                             dtype=af.Dtype.f64
                            )

        self.performance_test_flag = False

    _communicate_f = communicate_f

def test_periodic():

    obj = test('periodic', 'periodic')
    obj.f[obj.N_ghost:-obj.N_ghost,obj.N_ghost:-obj.N_ghost] = \
    af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)[obj.N_ghost:
                                                                  -obj.N_ghost,
                                                                  obj.N_ghost:
                                                                  -obj.N_ghost
                                                                 ]


    apply_bcs_f(obj)

    expected = af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)
    assert (af.max(af.abs(obj.f - expected)) < 5e-14)

def test_mirror():

    obj = test('mirror', 'mirror')
    obj.f[obj.N_ghost:-obj.N_ghost,obj.N_ghost:-obj.N_ghost] = \
    af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)[obj.N_ghost:
                                                                  -obj.N_ghost,
                                                                  obj.N_ghost:
                                                                  -obj.N_ghost
                                                                 ]


    apply_bcs_f(obj)

    expected = af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center)
    N_g      = obj.N_ghost

    expected[:N_g, :]  = af.flip(expected[N_g:2 * N_g], 0)
    expected[-N_g:, :] = af.flip(expected[-2 * N_g:-N_g], 0)
    expected[:, -N_g:] = af.flip(expected[:, -2 * N_g:-N_g], 1)
    expected[:, :N_g]  = af.flip(expected[:, N_g:2 * N_g], 1)

    assert (af.max(af.abs(obj.f - expected)) < 5e-14)

def test_dirichlet():
    
    obj = test('dirichlet', 'dirichlet')

    obj._A_q1, obj._A_q2 = 100, 100
    obj.dt               = 0.001

    obj.f = af.constant(0, obj.q1_center.shape[0], 
                        obj.q1_center.shape[1], 
                        obj.q1_center.shape[2],
                        dtype = af.Dtype.f64
                       )

    apply_bcs_f(obj)

    expected = af.constant(0, obj.q1_center.shape[0], 
                           obj.q1_center.shape[1], 
                           obj.q1_center.shape[2],
                           dtype = af.Dtype.f64
                          )

    N_g = obj.N_ghost

    expected[:N_g]     = af.select(obj.q1_center<obj.q1_start, 1, expected)[:N_g]
    expected[-N_g:]    = af.select(obj.q1_center>obj.q1_end, 1, expected)[-N_g:]
    expected[:, :N_g]  = af.select(obj.q2_center<obj.q2_start, 2, expected)[:, :N_g]
    expected[:, -N_g:] = af.select(obj.q2_center>obj.q2_end, 2, expected)[:, -N_g:]

    assert (af.max(af.abs(obj.f[:, N_g:-N_g] - expected[:, N_g:-N_g])) < 5e-14)
    assert (af.max(af.abs(obj.f[N_g:-N_g, :] - expected[N_g:-N_g, :])) < 5e-14)
