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
    import communicate_f, communicate_fields


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

        self.q1 = af.reorder(self.q1, 2, 0, 1)
        self.q2 = af.reorder(self.q2, 2, 0, 1)

        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof=(self.N_p1 * self.N_p2 * self.N_p3),
                                         stencil_width=self.N_ghost,
                                         boundary_type=('periodic', 'periodic'),
                                         stencil_type=1, 
                                        )

        self._glob_f  = self._da_f.createGlobalVec()
        self._local_f = self._da_f.createLocalVec()

        self._glob_f_array  = self._glob_f.getArray()
        self._local_f_array = self._local_f.getArray()

        self.boundary_conditions = type('obj', (object, ),
                                        {'in_q1':'periodic',
                                         'in_q2':'periodic'
                                        }
                                       )


        self.f = af.constant(0,
                             self.N_p1 * self.N_p2 * self.N_p3,
                             self.N_q1 + 2 * self.N_ghost,
                             self.N_q2 + 2 * self.N_ghost,
                             dtype=af.Dtype.f64
                            )

        self.f[:, self.N_ghost:-self.N_ghost,self.N_ghost:-self.N_ghost] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[:, self.N_ghost:
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

        N_g = self.N_ghost = np.random.randint(1, 5)

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

        self.q1 = af.reorder(self.q1, 2, 0, 1)
        self.q2 = af.reorder(self.q2, 2, 0, 1)

        self.q1 = af.tile(self.q1, 6)
        self.q2 = af.tile(self.q2, 6)

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

        self.cell_centered_EM_fields = af.constant(0, 6, self.q1.shape[1], 
                                                   self.q1.shape[2],
                                                   dtype=af.Dtype.f64
                                                  )

        self.cell_centered_EM_fields[:, N_g:-N_g, N_g:-N_g] = \
            af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)[:, N_g:-N_g,N_g:-N_g]
        
        self.performance_test_flag = False

def test_communicate_f():
    obj = test_distribution_function()
    communicate_f(obj)

    expected = af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)
    assert (af.mean(af.abs(obj.f - expected)) < 5e-14)

def test_communicate_fields():
    obj = test_fields()
    communicate_fields(obj)

    Ng = obj.N_ghost

    expected = af.sin(2 * np.pi * obj.q1 + 4 * np.pi * obj.q2)
    assert (af.mean(af.abs(obj.cell_centered_EM_fields - expected)) < 5e-14)
