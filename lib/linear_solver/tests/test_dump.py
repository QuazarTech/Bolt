#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file contains the unit tests to ensure proper functioning of the
# function under lib/linear_solver/dump.py. A test file test_file.h5 is
# created in the process. It is ensured that the read and write processes
# are carried out as expected.

# Importing dependencies:
import numpy as np
import arrayfire as af
import h5py

# Importing Solver functions:
from lib.linear_solver.dump import dump_variables, dump_distribution_function


class test(object):
    def __init__(self):
        self.N_q1 = 2
        self.N_q2 = 3
        self.N_p1 = 4
        self.N_p2 = 5
        self.N_p3 = 6

        self.f_expand = np.random.rand(2, 3, 4, 5, 6)
        self.f = af.to_array(self.f_expand.reshape(2, 3, 4 * 5 * 6))


def test_dump_variables():
    a = af.randu(10, dtype=af.Dtype.c64)
    b = af.randu(10, 10, dtype=af.Dtype.c64)
    c = af.randu(10, dtype=af.Dtype.f64)
    d = af.randu(10, 10, dtype=af.Dtype.f64)
    dump_variables(test, 'test_file', a, b, c, d, 'a', 'b', 'c', 'd')

    h5f = h5py.File('test_file.h5', 'r')
    a_read = h5f['a'][:]
    b_read = h5f['b'][:]
    c_read = h5f['c'][:]
    d_read = h5f['d'][:]
    h5f.close()

    check = (a_read - np.array(a)) + (b_read - np.array(b)) + \
        (c_read - np.array(c)) + (d_read - np.array(d))
    assert(np.all(check == 0))


def test_dump_distribution_function():
    test_obj = test()
    dump_distribution_function(test_obj, 'test_file')

    h5f = h5py.File('test_file.h5', 'r')
    f_read = h5f['distribution_function'][:]
    h5f.close()

    assert(np.all(test_obj.f_expand - f_read == 0))
