#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
from petsc4py import PETSc
import h5py


def dump_moments(self, file_name):
    """
    This function is used to dump variables to a file for later usage.

    Parameters
    ----------

    file_name : str
                The variables will be dumped to this provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing all the moments that were defined under moments_defs in
    physical_system.

    Examples
    --------

    >> solver.dump_variables('boltzmann_moments_dump')

    The above set of statements will create a HDF5 file which contains the
    all the moments which have been defined in the physical_system object.
    The data is always stored with the key 'moments' inside the HDF5 file.
    Suppose 'density' and 'energy' are two these moments, and are declared
    the first and second in the moment_exponents object:

    These variables can then be accessed from the file using
    
    >> import h5py
    
    >> h5f = h5py.File('boltzmann_moments_dump.h5', 'r')
    
    >> rho = h5f['moments'][:][:, :, 0]
    
    >> E   = h5f['moments'][:][:, :, 1]
    
    >> h5f.close()
    """
    i = 0
    
    for key in self.physical_system.moment_exponents:
        self._glob_moments_value[:][:, :, i] = \
        np.array(self.compute_moments(key))
        i += 1
    
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w')
    viewer(self._glob_moments)


def dump_distribution_function(self, file_name):
    """
    This function is used to dump distribution function to a file for
    later usage.This dumps the complete 5D distribution function which
    can be used for post-processing

    Parameters
    ----------

    file_name : The distribution_function array will be dumped to this
                provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing the data of the distribution function

    Examples
    --------
    
    >> solver.dump_distribution_function('distribution_function')

    The above statement will create a HDF5 file which contains the
    distribution function. The data is always stored with the key 
    'distribution_function'

    This can later be accessed using

    >> import h5py
    
    >> h5f = h5py.File('distribution_function', 'r')
    
    >> f   = h5f['distribution_function'][:]
    
    >> h5f.close()
    """
    self._glob_f_value[:] =   0.5 * self.N_q2 * self.N_q1
                            * np.array(af.ifft2(self.Y[:, :, :, 0])).real
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w')
    viewer(self._glob_f)
