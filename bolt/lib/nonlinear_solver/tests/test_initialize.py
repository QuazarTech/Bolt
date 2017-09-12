#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checks that the initialize function initializes the
distribution function array as we expect.
"""


# Importing dependencies:
import numpy as np
from numpy.fft import fftfreq
import arrayfire as af
from petsc4py import PETSc

# Importing solver functions:
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver
from bolt.lib.nonlinear_solver.compute_moments \
    import compute_moments as compute_moments_imported
from bolt.lib.nonlinear_solver.communicate import communicate_fields

initialize = nonlinear_solver._initialize

moment_exponents = dict(density     = [0, 0, 0],
                        mom_p1_bulk = [1, 0, 0],
                        mom_p2_bulk = [0, 1, 0],
                        mom_p3_bulk = [0, 0, 1],
                        energy      = [2, 2, 2]
                        )

moment_coeffs = dict(density     = [1, 0, 0],
                     mom_p1_bulk = [1, 0, 0],
                     mom_p2_bulk = [0, 1, 0],
                     mom_p3_bulk = [0, 0, 1],
                     energy      = [1, 1, 1]
                    )

def MB_dist(q1, q2, p1, p2, p3, params):

    # Calculating the perturbed density:
    rho = 1 + af.cos(2 * np.pi * q1 + 4 * np.pi * q2)

    f = rho * (1 / (2 * np.pi))**(3 / 2) \
            * af.exp(-0.5 * p1**2) \
            * af.exp(-0.5 * p2**2) \
            * af.exp(-0.5 * p3**2)

    af.eval(f)
    return (f)

class params:
    def __init__(self):
        pass

class test(object):
    def __init__(self):
        # Initializing object with required parameters:
        self.physical_system = type('obj', (object, ),
                                    {'initial_conditions':
                                      type('obj', (object,), {'initialize_f':MB_dist}),
                                     'params':
                                      type('obj', (object,), {'fields_initialize':'fft',
                                                              'charge_electron':-1
                                                              }),
                                      'moment_exponents': moment_exponents,
                                      'moment_coeffs': moment_coeffs
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

        self.p1_start = np.random.randint(-10, -5)
        self.p2_start = np.random.randint(-10, -5)
        self.p3_start = np.random.randint(-10, -5)

        self.p1_end = np.random.randint(5, 10)
        self.p2_end = np.random.randint(5, 10)
        self.p3_end = np.random.randint(5, 10)

        self.N_p1 = np.random.randint(16, 32)
        self.N_p2 = np.random.randint(16, 32)
        self.N_p3 = np.random.randint(16, 32)

        self.dp1 = (self.p1_end - self.p1_start) / self.N_p1
        self.dp2 = (self.p2_end - self.p2_start) / self.N_p2
        self.dp3 = (self.p3_end - self.p3_start) / self.N_p3

        self.N_ghost = 1

        self._comm = PETSc.COMM_WORLD
        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof = 1,
                                         stencil_width=self.N_ghost
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

        self._local_value_fields = self._da_fields.getVecArray(self._local_fields)
        self._glob_value_fields  = self._da_fields.getVecArray(self._glob_fields)
        
        self.q1_center, self.q2_center = nonlinear_solver._calculate_q_center(self)
        self.p1, self.p2, self.p3      = nonlinear_solver._calculate_p_center(self)

    compute_moments     = compute_moments_imported
    _communicate_fields = communicate_fields

def test_initialize():
    obj = test()
    initialize(obj, params)

    q1 = af.tile(obj.q1_center, 1, 1, obj.N_p1*obj.N_p2*obj.N_p3)
    q2 = af.tile(obj.q2_center, 1, 1, obj.N_p1*obj.N_p2*obj.N_p3)

    rho = 1 + af.cos(2 * np.pi * q1 + 4 * np.pi * q2)

    p1 = af.tile(obj.p1, obj.N_q1 + 2, obj.N_q2 + 2)
    p2 = af.tile(obj.p2, obj.N_q1 + 2, obj.N_q2 + 2)
    p3 = af.tile(obj.p3, obj.N_q1 + 2, obj.N_q2 + 2)

    f_ana = rho * (1 / (2 * np.pi))**(3 / 2) \
                * af.exp(-0.5 * p1**2) \
                * af.exp(-0.5 * p2**2) \
                * af.exp(-0.5 * p3**2)

    assert(af.sum(af.abs(obj.f - f_ana))<1e-13)
