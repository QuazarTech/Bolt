#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The communicate routines are used to communicate the information
about the edge zones amongst the different processors when run
in parallel. Additionally, it applies periodic boundary conditions
automatically.

While this test doesn't currently check the operation when run in
parallel, it ensures that the periodic boundary conditions are
applied correctly.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.communicate \
    import communicate_distribution_function, communicate_fields


class test_distribution_function(object):
    def __init__(self):
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

        self.q1 =   self.q1_start \
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q1 + self.N_ghost
                                    )
                    ) * self.dq1

        self.q2 =   self.q2_start \
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q2 + self.N_ghost
                                    )
                    ) * self.dq2

        self.q1 = af.tile(af.to_array(self.q1), 1,
                          self.N_q2 + 2 * self.N_ghost,
                          self.N_p1 * self.N_p2 * self.N_p3
                         )

        self.q2 = af.tile(af.reorder(af.to_array(self.q2)),
                          self.N_q1 + 2 * self.N_ghost, 1,
                          self.N_p1 * self.N_p2 * self.N_p3
                         )

        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof=(self.N_p1 * self.N_p2 * self.N_p3),
                                         stencil_width=self.N_ghost,
                                         boundary_type=('periodic', 'periodic'),
                                         stencil_type=1, 
                                        )

        self._glob_f  = self._da_f.createGlobalVec()
        self._local_f = self._da_f.createLocalVec()

        self._glob_value_f  = self._da_f.getVecArray(self._glob_f)
        self._local_value_f = self._da_f.getVecArray(self._local_f)

        self.f = af.constant(0,
                             self.N_q1,
                             self.N_q2,
                             self.q1.shape[2],
                             dtype=af.Dtype.f64
                            )

        self.f[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]
        self.performance_test_flag = False


class test_fields(object):
    def __init__(self):
        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = np.random.randint(16, 32)
        self.N_q2 = np.random.randint(16, 32)

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = np.random.randint(1, 5)

        self.q1 = self.q1_start \
                  * (0.5 + np.arange(-self.N_ghost,
                                     self.N_q1 + self.N_ghost
                                    )
                    ) * self.dq1

        self.q2 = self.q2_start \
                  * (0.5 + np.arange(-self.N_ghost,
                                    self.N_q2 + self.N_ghost
                                    )
                    ) * self.dq2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
        self.q1, self.q2 = af.to_array(self.q1), af.to_array(self.q2)

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

        self.E1 = af.constant(0, self.q1.shape[0], self.q1.shape[1],
                              dtype=af.Dtype.f64
                             )

        self.E2 = self.E1.copy()
        self.E3 = self.E1.copy()
        self.B1 = self.E1.copy()
        self.B2 = self.E1.copy()
        self.B3 = self.E1.copy()

        self.E1[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]

        self.E2[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]
        
        self.E3[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]

        self.B1[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]

        self.B2[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]

        self.B3[self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[self.N_ghost:
                                                              -self.N_ghost,
                                                              self.N_ghost:
                                                              -self.N_ghost
                                                             ]

def test_communicate_distribution_function():
    obj = test_distribution_function()
    communicate_distribution_function(obj)

    expected = af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)

    assert (af.max(af.abs(obj.f - expected)) < 5e-14)


def test_communicate_fields():
    obj = test_fields()
    communicate_fields(obj)

    expected = af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)
    assert (af.max(af.abs(obj.E1 - expected)) < 5e-14)
    assert (af.max(af.abs(obj.E2 - expected)) < 5e-14)
    assert (af.max(af.abs(obj.E3 - expected)) < 5e-14)
    assert (af.max(af.abs(obj.B1 - expected)) < 5e-14)
    assert (af.max(af.abs(obj.B2 - expected)) < 5e-14)
    assert (af.max(af.abs(obj.B3 - expected)) < 5e-14)
