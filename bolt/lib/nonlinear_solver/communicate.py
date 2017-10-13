#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

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
    
    N_g = self.N_ghost

    self._local_value_f[:] = np.array(self.f)
    # Global value is non-inclusive of the ghost-zones:
    self._glob_value_f[:] = (self._local_value_f[:])[N_g:-N_g, N_g:-N_g]
    
    # The following function takes care of periodic boundary conditions,
    # and interzonal communications:
    self._da_f.globalToLocal(self._glob_f, self._local_f)

    # Converting back from PETSc.Vec to af.Array:
    self.f = af.to_array(self._local_value_f[:])

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_communicate_f += toc - tic
    
    return


def communicate_fields(self, on_fdtd_grid=False):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the EM field
    arrays.
    """
    if(self.performance_test_flag == True):
        tic = af.time()
    
    N_ghost = self.N_ghost

    # Assigning the values of the af.Array fields quantities
    # to the PETSc.Vec:

    if(on_fdtd_grid is True):
        (self._local_value_fields[:])[:, :, 0] = np.array(self.E1_fdtd)
        (self._local_value_fields[:])[:, :, 1] = np.array(self.E2_fdtd)
        (self._local_value_fields[:])[:, :, 2] = np.array(self.E3_fdtd)

        (self._local_value_fields[:])[:, :, 3] = np.array(self.B1_fdtd)
        (self._local_value_fields[:])[:, :, 4] = np.array(self.B2_fdtd)
        (self._local_value_fields[:])[:, :, 5] = np.array(self.B3_fdtd)

    else:
        (self._local_value_fields[:])[:, :, 0] = np.array(self.E1)
        (self._local_value_fields[:])[:, :, 1] = np.array(self.E2)
        (self._local_value_fields[:])[:, :, 2] = np.array(self.E3)

        (self._local_value_fields[:])[:, :, 3] = np.array(self.B1)
        (self._local_value_fields[:])[:, :, 4] = np.array(self.B2)
        (self._local_value_fields[:])[:, :, 5] = np.array(self.B3)

    # Global value is non-inclusive of the ghost-zones:
    self._glob_value_fields[:] = (self._local_value_fields[:])[N_ghost:-N_ghost,
                                                               N_ghost:-N_ghost,
                                                               :
                                                               ]

    # Takes care of boundary conditions and interzonal communications:
    self._da_fields.globalToLocal(self._glob_fields, self._local_fields)

    # Converting back to af.Array
    if(on_fdtd_grid is True):
        self.E1_fdtd = af.to_array((self._local_value_fields[:])[:, :, 0])
        self.E2_fdtd = af.to_array((self._local_value_fields[:])[:, :, 1])
        self.E3_fdtd = af.to_array((self._local_value_fields[:])[:, :, 2])

        self.B1_fdtd = af.to_array((self._local_value_fields[:])[:, :, 3])
        self.B2_fdtd = af.to_array((self._local_value_fields[:])[:, :, 4])
        self.B3_fdtd = af.to_array((self._local_value_fields[:])[:, :, 5])
        
        af.eval(self.E1_fdtd, self.E2_fdtd, self.E3_fdtd,
                self.B1_fdtd, self.B2_fdtd, self.B3_fdtd
               )

    else:
        self.E1 = af.to_array((self._local_value_fields[:])[:, :, 0])
        self.E2 = af.to_array((self._local_value_fields[:])[:, :, 1])
        self.E3 = af.to_array((self._local_value_fields[:])[:, :, 2])

        self.B1 = af.to_array((self._local_value_fields[:])[:, :, 3])
        self.B2 = af.to_array((self._local_value_fields[:])[:, :, 4])
        self.B3 = af.to_array((self._local_value_fields[:])[:, :, 5])

        af.eval(self.E1, self.E2, self.E3, self.B1, self.B2, self.B3)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_communicate_fields += toc - tic

    return
