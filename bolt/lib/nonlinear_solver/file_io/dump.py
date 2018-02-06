#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import arrayfire as af

def dump_aux_arrays(self, arrays, name, file_name):

    if (self.dump_aux_arrays_initial_call):
        self._da_aux_arrays = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                   dof        = len(arrays),
                                                   proc_sizes = (PETSc.DECIDE,
                                                                 PETSc.DECIDE
                                                                ),
                                                   comm       = self._comm
                                                 )
    
        self._glob_aux       = self._da_aux_arrays.createGlobalVec()
        self._glob_aux_array = self._glob_aux.getArray()

        self.dump_aux_arrays_initial_call = 0

    N_g = self.N_ghost

    for i in range(len(arrays)):
        if (i==0):
            array_to_dump = arrays[0][:, N_g:-N_g, N_g:-N_g]
        else:
            array_to_dump = af.join(0, array_to_dump,
                                    arrays[i][:, N_g:-N_g, N_g:-N_g]
                                   )

    af.flat(array_to_dump).to_ndarray(self._glob_aux_array)
    PETSc.Object.setName(self._glob_aux, name)
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
    viewer(self._glob_aux)

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
    N_g = self.N_ghost

    i = 0
    for key in self.physical_system.moment_exponents:
        if(i == 0):
            array_to_dump = self.compute_moments(key)[:, N_g:-N_g,N_g:-N_g]
        else:
            array_to_dump = af.join(0, array_to_dump,
                                    self.compute_moments(key)[:, N_g:-N_g,N_g:-N_g]
                                   )
        i += 1

    af.flat(array_to_dump).to_ndarray(self._glob_moments_array)
    PETSc.Object.setName(self._glob_moments, 'moments')
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
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
    N_g = self.N_ghost
    
    af.flat(self.f[:, N_g:-N_g, N_g:-N_g]).to_ndarray(self._glob_f_array)
    PETSc.Object.setName(self._glob_f, 'distribution_function')
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
    viewer(self._glob_f)

    return
