#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
from bolt.lib.utils.af_petsc_conversion import af_to_petsc_glob_array
from bolt.lib.utils.af_petsc_conversion import petsc_local_array_to_af

def communicate_f(self):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the distribution
    function array.
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    # Transfer data from af.Array to PETSc.Vec
    af_to_petsc_glob_array(self, self.f, self._glob_f_array)

    # The following function takes care of interzonal communications
    # Additionally, it also automatically applies periodic BCs when necessary
    self._da_f.globalToLocal(self._glob_f, self._local_f)

    # Converting back from PETSc.Vec to af.Array:
    self.f = petsc_local_array_to_af(self, 
                                     self.N_p1*self.N_p2*self.N_p3,
                                     self.N_species,
                                     self._local_f_array
                                    )

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_communicate_f += toc - tic

    return

def communicate_fields(self, on_fdtd_grid = False):
    """
    Used in communicating the values at the boundary zones for each of
    the local vectors among all procs.This routine is called to take care
    of communication(and periodic B.C's) procedures for the EM field
    arrays. The function is used for communicating the EM field values 
    on the cell centered grid  which is used by default. Additionally,it can
    also be used to communicate the values on the Yee-grid which is used by the FDTD solver.
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    # Assigning the values of the af.Array 
    # fields quantities to the PETSc.Vec:
    if(on_fdtd_grid is True):
        tmp_array = self.yee_grid_EM_fields
    else:
        tmp_array = self.cell_centered_EM_fields

    af_to_petsc_glob_array(self, tmp_array, self._glob_fields_array)

    # Takes care of boundary conditions and interzonal communications:
    self._da_fields.globalToLocal(self._glob_fields, self._local_fields)

    # Converting back to af.Array
    tmp_array = petsc_local_array_to_af(self, 6, 1, self._local_fields_array)

    if(on_fdtd_grid is True):
        self.yee_grid_EM_fields = tmp_array
    else:
        self.cell_centered_EM_fields = tmp_array
        
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_communicate_fields += toc - tic

    return
