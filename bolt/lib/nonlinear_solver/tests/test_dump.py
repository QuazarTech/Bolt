#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the unit tests to ensure proper functioning of the
functions under lib/nonlinear_solver/dump.py. A test file test_file.h5 is
created in the process. It is ensured that the read and write processes
are carried out as expected.
"""

# Importing dependencies:
import numpy as np
import arrayfire as af
import h5py
from petsc4py import PETSc

# Importing Solver functions:
from bolt.lib.nonlinear_solver.dump \
    import dump_moments, dump_distribution_function

from bolt.lib.nonlinear_solver.compute_moments import \
    compute_moments as compute_moments_imported

from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

calculate_p = nonlinear_solver._calculate_p_center    

moment_exponents = dict(density = [0, 0, 0],
                        energy  = [2, 2, 2]
                        )

moment_coeffs = dict(density = [1, 0, 0],
                     energy  = [1, 1, 1]
                     )

class test(object):
    
    def __init__(self):
        self.N_p1 = 2
        self.N_p2 = 3
        self.N_p3 = 4
        self.N_q1 = 5
        self.N_q2 = 6

        self.N_ghost = 1

        self.p1_start = self.p2_start = self.p3_start = 0

        self.dp1 = 2/self.N_p1
        self.dp2 = 2/self.N_p2
        self.dp3 = 2/self.N_p3

        self.p1, self.p2, self.p3 = self._calculate_p_center()

        self.physical_system = type('obj', (object,),
                                    {'moment_exponents': moment_exponents,
                                     'moment_coeffs':    moment_coeffs
                                    }
                                   )

        self.f = af.randu(self.N_q1 + 2 * self.N_ghost,
                          self.N_q2 + 2 * self.N_ghost,
                          self.N_p1 * self.N_p2 * self.N_p3,
                          dtype = af.Dtype.f64
                         )

        self._comm = PETSc.COMM_WORLD

        self._da_dump_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof=(  self.N_p1 
                                                   * self.N_p2
                                                   * self.N_p3
                                                  ),
                                              stencil_width = self.N_ghost
                                             )

        self._da_dump_moments = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                    dof=len(self.physical_system.\
                                                            moment_exponents
                                                           ),
                                                    stencil_width = self.N_ghost
                                                   )

        self._glob_f       = self._da_dump_f.createGlobalVec()
        self._glob_f_value = self._da_dump_f.getVecArray(self._glob_f)

        self._glob_moments       = self._da_dump_moments.createGlobalVec()
        self._glob_moments_value = self._da_dump_moments.\
                                   getVecArray(self._glob_moments)

        PETSc.Object.setName(self._glob_f, 'distribution_function')
        PETSc.Object.setName(self._glob_moments, 'moments')
    
    compute_moments     = compute_moments_imported
    _calculate_p_center = calculate_p

def test_dump_distribution_function():
    test_obj                  = test()
    N_g                       = test_obj.N_ghost
    test_obj._glob_f_value[:] = np.array(test_obj.f)[N_g:-N_g, 
                                                     N_g:-N_g
                                                    ]
    
    dump_distribution_function(test_obj, 'test_file')

    h5f    = h5py.File('test_file.h5', 'r')
    f_read = h5f['distribution_function'][:]
    h5f.close()

    f_read_reordered = np.swapaxes(f_read, 0, 1)

    assert(np.sum(abs(  f_read_reordered 
                      - np.array(test_obj.f[N_g:-N_g, N_g:-N_g])))\
           /f_read_reordered.size<1e-14
          )

def test_dump_moments():
    test_obj = test()
    N_g      = test_obj.N_ghost

    dump_moments(test_obj, 'test_file')

    h5f          = h5py.File('test_file.h5', 'r')
    moments_read = h5f['moments'][:]
    h5f.close()

    moments_read = np.swapaxes(moments_read, 0, 1)

    assert(af.sum(af.to_array(moments_read[:, :, 0]) - 
                  compute_moments_imported(test_obj, 'density')[N_g:-N_g, N_g:-N_g]
                 )==0
          )

    assert(af.sum(af.to_array(moments_read[:, :, 1]) - 
                  compute_moments_imported(test_obj, 'energy')[N_g:-N_g, N_g:-N_g]
                 )==0
          )
