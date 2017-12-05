#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the unit tests to ensure proper functioning of the
function under lib/linear_solver/dump.py. A test file test_file.h5 is
created in the process. It is ensured that the read and write processes
are carried out as expected.
"""

# Importing dependencies:
import numpy as np
import arrayfire as af

import h5py
from petsc4py import PETSc

# Importing Solver functions:
from bolt.lib.linear_solver.file_io.dump \
    import dump_moments, dump_distribution_function
from bolt.lib.linear_solver.file_io.load \
    import load_distribution_function

from bolt.lib.linear_solver.compute_moments import \
    compute_moments as compute_moments_imported

from bolt.lib.linear_solver.linear_solver import linear_solver
calculate_p = linear_solver._calculate_p_center    

moment_exponents = dict(density = [0, 0, 0],
                        energy  = [2, 2, 2]
                       )

moment_coeffs = dict(density = [1, 0, 0],
                     energy  = [1, 1, 1]
                    )

class test(object):
    
    def __init__(self):
        self.N_q1 = 32
        self.N_q2 = 32
        self.N_p1 = 4
        self.N_p2 = 5
        self.N_p3 = 6

        self.p1_start = self.p2_start = self.p3_start = 0

        self.dp1 = 2/self.N_p1
        self.dp2 = 2/self.N_p2
        self.dp3 = 2/self.N_p3

        self.p1, self.p2, self.p3 = self._calculate_p_center()

        # Creating an object with required attributes for the test:
        self.physical_system = type('obj', (object,),
                                    {'moment_exponents': moment_exponents,
                                     'moment_coeffs':    moment_coeffs
                                    }
                                   )

        self.single_mode_evolution = False

        self.f = af.randu(self.N_q1, self.N_q2,
                          self.N_p1 * self.N_p2 * self.N_p3,
                          dtype = af.Dtype.f64
                         )

        self.Y = 2 * af.fft2(self.f)/(self.N_q1 * self.N_q2)

        self._da_dump_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof = (  self.N_p1 
                                                     * self.N_p2 
                                                     * self.N_p3
                                                    ),
                                             )

        self._da_dump_moments = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                    dof = len(self.physical_system.\
                                                              moment_exponents)
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
    test_obj = test()

    f_before_load = test_obj.Y.copy()

    dump_distribution_function(test_obj, 'test_file')
    load_distribution_function(test_obj, 'test_file')

    assert(af.mean(af.abs(test_obj.Y - f_before_load)) < 1e-14)

def test_dump_moments():
    test_obj = test()
    dump_moments(test_obj, 'test_file')

    h5f = h5py.File('test_file.h5', 'r')
    moments_read = h5f['moments'][:]
    h5f.close()

    moments_read = np.swapaxes(moments_read, 0, 1)

    assert(af.sum(af.to_array(moments_read[:, :, 0]) - 
                  compute_moments_imported(test_obj, 'density')
                 )==0
          )

    assert(af.sum(af.to_array(moments_read[:, :, 1]) - 
                  compute_moments_imported(test_obj, 'energy')
                 )==0
          )
