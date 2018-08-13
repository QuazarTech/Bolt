#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import h5py
import arrayfire as af

def dump_moments(self, file_name):
    """
    This function is used to dump moment variables to a file for later usage.

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
    Suppose 'density', 'mom_v1_bulk' and 'energy' are the 3 functions defined.
    Then the moments get stored in alphabetical order, ie. 'density', 'energy'...:

    These variables can then be accessed from the file using
    
    >> import h5py
    
    >> h5f    = h5py.File('boltzmann_moments_dump.h5', 'r')
    
    >> n      = h5f['moments'][:][:, :, 0]
    
    >> energy = h5f['moments'][:][:, :, 1]
    
    >> mom_v1 = h5f['moments'][:][:, :, 2]
    
    >> h5f.close()

    However, in the case of multiple species, the following holds:

    >> n_species_1      = h5f['moments'][:][:, :, 0]
 
    >> n_species_2      = h5f['moments'][:][:, :, 1]
    
    >> energy_species_1 = h5f['moments'][:][:, :, 2]

    >> energy_species_2 = h5f['moments'][:][:, :, 3]
    
    >> mom_v1_species_1 = h5f['moments'][:][:, :, 4]

    >> mom_v1_species_2 = h5f['moments'][:][:, :, 5]
    """
    N_g = self.N_ghost

    attributes = [a for a in dir(self.physical_system.moments) if not a.startswith('_')]

    # Removing utility functions:
    if('integral_over_v' in attributes):
        attributes.remove('integral_over_v')

    for i in range(len(attributes)):
        if(i == 0):
            array_to_dump = self.compute_moments(attributes[i])[:, :, N_g:-N_g,N_g:-N_g]
        else:
            array_to_dump = af.join(1, array_to_dump,
                                    self.compute_moments(attributes[i])[:, :, N_g:-N_g, N_g:-N_g]
                                   )

        af.eval(array_to_dump)

    af.flat(array_to_dump).to_ndarray(self._glob_moments_array)
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
    viewer(self._glob_moments)

    # Following segment shows discrepancy due to buggy behaviour of af.mean
    # Reported in https://github.com/QuazarTech/Bolt/issues/46
    # print("MEAN_n for species 1(RAW DATA):", af.mean(array_to_dump[:, 0, :, :]))
    # print("MEAN_n for species 2(RAW DATA):", af.mean(array_to_dump[:, 1, :, :]))

    # h5f = h5py.File(file_name + '.h5', 'r')
    # mom = np.swapaxes(h5f['moments'][:], 0, 1)
    # h5f.close()

    # print("MEAN_n for species 1(DUMP DATA):", np.mean(mom[:, :, 0]))
    # print("MEAN_n for species 2(DUMP DATA):", np.mean(mom[:, :, 1]))

def dump_distribution_function(self, file_name):
    """
    This function is used to dump distribution function to a file for
    later usage.This dumps the complete 5D distribution function which
    can be used for restarting / post-processing

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
    
    >> h5f = h5py.File('distribution_function.h5', 'r')
    
    >> f   = h5f['distribution_function'][:]
    
    >> h5f.close()

    Alternatively, it can also be used with the load function to resume
    a long-running calculation.

    >> solver.load_distribution_function('distribution_function')
    """
    N_g = self.N_ghost
    
    N_q1_local = self.f.shape[2]
    N_q2_local = self.f.shape[3]

    array_to_dump = self.f
    
    array_to_dump = af.flat(array_to_dump[:, :, N_g:-N_g, N_g:-N_g])
    array_to_dump.to_ndarray(self._glob_f_array)
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
    viewer(self._glob_f)

    return

def dump_EM_fields(self, file_name):
    """
    This function is used to EM fields to a file for later usage.
    This dumps all the EM fields quantities E1, E2, E3, B1, B2, B3 
    which can then be used later for post-processing

    Parameters
    ----------

    file_name : The EM_fields array will be dumped to this
                provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing the data of the EM fields.

    Examples
    --------
    
    >> solver.dump_EM_fields('data_EM_fields')

    The above statement will create a HDF5 file which contains the
    EM fields data. The data is always stored with the key 
    'EM_fields'

    This can later be accessed using

    >> import h5py
    
    >> h5f = h5py.File('data_EM_fields.h5', 'r')
    
    >> EM_fields = h5f['EM_fields'][:]

    >> E1 = EM_fields[:, :, 0]
    
    >> E2 = EM_fields[:, :, 1]
    
    >> E3 = EM_fields[:, :, 2]
    
    >> B1 = EM_fields[:, :, 3]
    
    >> B2 = EM_fields[:, :, 4]
    
    >> B3 = EM_fields[:, :, 5]

    >> h5f.close()

    Alternatively, it can also be used with the load function to resume
    a long-running calculation.

    >> solver.load_EM_fields('data_EM_fields')
    """
    N_g = self.N_ghost
    
    flattened_global_EM_fields_array = \
        af.flat(self.fields_solver.yee_grid_EM_fields[:, :, N_g:-N_g, N_g:-N_g])
    flattened_global_EM_fields_array.to_ndarray(self.fields_solver._glob_fields_array)
    
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
    viewer(self.fields_solver._glob_fields)

    return
