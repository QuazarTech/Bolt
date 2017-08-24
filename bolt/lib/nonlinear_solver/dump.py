#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc


# def dump_variables(file_name, self, **args):
#     """
#     This function is used to dump variables to a file for later usage.

#     Parameters:
#     -----------
#     This function takes the variable names which need to be dumped followed
#     by their global vectors and the keys with which the variables are stored.

#     Output:
#     -------
#     This function returns None. However it creates a file 'glob_vector.h5',
#     containing the variables passed to this function.

#     Example:
#     --------
#     >> density     = solver.compute_moments('density')
#     >> temperature = solver.compute_moments('energy')/density
#     >> solver.dump_variables(density, temperature, density_glob,
#                              temperature_glob,'density', 'temperature')

#     The above set of statements will create 2 HDF5 files which contains the
#     density, and temperature stored under the keys of 'density' and
#     'temperature'

#     These variables can then be accessed from the file using:
#     >> import h5py
#     >> h5f = h5py.File('density_glob.h5', 'r')
#     >> rho = h5f['density'][:]
#     >> h5f.close()
#     """
#     h5f = h5py.File(file_name + '.h5', 'w')
#     for i in range(int(len(args) / 3)):
#         PETSc.Object.setName(self._glob, 'distribution_function')
#         viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm=self._comm)
#         global_vec_value = self._da.getVecArray(self.glob)

#         h5f.create_dataset(args[i + int(len(args) / 2)], data=args[i])
#     h5f.close()
#     return


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
    viewer(self._glob)

    return
