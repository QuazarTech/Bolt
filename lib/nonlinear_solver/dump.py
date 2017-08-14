#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from petsc4py import PETSc


def dump_variables(self, file_name, **args):
    """
    This function is used to dump variables to a file for later usage.

    Parameters:
    -----------
    file_name : The variables will be dumped to this provided file name.

    This is followed by the variable names which need to be dumped followed
    by the keys with which the variables are stored.

    Output:
    -------
    This function returns None. However it creates a file 'file_name.h5',
    containing the variables passed to this function.

    Example:
    --------
    >> density     = solver.compute_moments('density')
    >> temperature = solver.compute_moments('energy')/density
    >> solver.dump_variables('moments', density, temperature,\
                             'density', 'temperature')

    The above set of statements will create a HDF5 file which contains the
    density, and temperature stored under the keys of 'density' and
    'temperature'

    These variables can then be accessed from the file using:
    >> import h5py
    >> h5f = h5py.File('moments.h5', 'r')
    >> rho = h5f['density'][:]
    >> T   = h5f['temperature'][:]
    >> h5f.close()
    """
    h5f = h5py.File(file_name + '.h5', 'w')
    for i in range(int(len(args) / 2)):
        h5f.create_dataset(args[i + int(len(args) / 2)], data=args[i])
    h5f.close()
    return


def dump_distribution_function(self, file_name):
    """
    This function is used to dump distribution function to a file for
    later usage.This dumps the complete 5D distribution function which
    can be used for post-processing

    Parameters:
    -----------
    file_name : The distribution_function array will be dumped to this
                provided file name.

    Output:
    -------
    This function returns None. However it creates a file 'file_name.h5',
    containing the data of the distribution function

    Example:
    --------
    >> solver.dump_distribution_function('distribution_function')
    The above statement will create a HDF5 file which contains the
    distribution function.

    This can later be accessed using:
    >> import h5py
    >> h5f = h5py.File('distribution_function', 'r')
    >> f   = h5f['distribution_function'][:]
    >> h5f.close()

    """
    PETSc.Object.setName(self._glob, 'distribution_function')
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)

    global_vec_value = self._da.getVecArray(self._glob)
    global_vec_value[:] = np.array(self.f[self.N_ghost:-self.N_ghost,
                                          self.N_ghost:-self.N_ghost, :])
    viewer(self._glob)

    # creating the dump file using viewer however causes a reordering in the
    # original distribution function. We reopen this distribution file, and
    # rewrite to overcome this:
    # need to add this in the post processing file?

    h5f = h5py.File('file_name' + '.h5', 'r')
    f = h5f['distribution_function'][:]
    h5f.close()

    f = np.swapaxes(f, 0, 1).reshape(self.N_q1, self.N_q2,
                                     self.N_p3, self.N_p2, self.N_p1)
    f = np.swapaxes(f, 4, 2)

    h5f = h5py.File('file_name' + '.h5', 'r')
    h5f.create_dataset('distribution_function', data=f)
    h5f.close()

    return
