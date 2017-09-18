#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test we check that the 2D Poisson solver
works as intended. For this purpose, we assign
a density distribution for which the analytical
solution for electrostatic fields may be computed.

This solution is then checked against the solution
given by the KSP solver
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields


class test(object):
    def __init__(self, N):
        # Creating an object with necessary attributes:
        self.physical_system = type('obj', (object, ),
                                    {'params': type('obj', (object, ),
                                        {'charge_electron': -1})
                                    }
                                   )

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
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q1 + self.N_ghost
                                    )
                    ) * self.dq1
        
        self.q2 = self.q2_start \
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q2 + self.N_ghost
                                    )
                    ) * self.dq2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
        self.q2, self.q1 = af.to_array(self.q2), af.to_array(self.q1)

        self.E1 = af.constant(0, self.q1.shape[0], self.q1.shape[1],
                              dtype=af.Dtype.f64
                             )

        self.E2 = self.E1.copy()
        self.E3 = self.E1.copy()

        self.B1 = self.E1.copy()
        self.B2 = self.E1.copy()
        self.B3 = self.E1.copy()

        self._comm = PETSc.COMM_WORLD.tompi4py()

        self._da_ksp = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                            stencil_width=self.N_ghost,
                                            boundary_type=('periodic',
                                                           'periodic'),
                                            stencil_type=1,
                                          )

    def compute_moments(self, *args):
        return (1 + af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))


def test_compute_electrostatic_fields():

    error_E1 = np.zeros(5)
    error_E2 = np.zeros(5)

    N = 2**np.arange(5, 10)

    for i in range(N.size):
        obj = test(N[i])
        compute_electrostatic_fields(obj)

        E1_expected =    (0.1 / np.pi) \
                       * af.cos(  2 * np.pi * obj.q1
                                + 4 * np.pi * obj.q2
                               )

        E2_expected =   (0.2 / np.pi) \
                      * af.cos(  2 * np.pi * obj.q1
                               + 4 * np.pi * obj.q2
                              )

        N_g = obj.N_ghost

        error_E1[i] = af.sum(af.abs(  obj.E1[N_g:-N_g, N_g:-N_g]
                                    - E1_expected[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (obj.E1[N_g:-N_g, N_g:-N_g].elements())

        error_E2[i] = af.sum(af.abs(  obj.E2[N_g:-N_g, N_g:-N_g]
                                    - E2_expected[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (obj.E2[N_g:-N_g, N_g:-N_g].elements())

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)

    assert (abs(poly_E1[0] + 2) < 0.2)
    assert (abs(poly_E2[0] + 2) < 0.2)
