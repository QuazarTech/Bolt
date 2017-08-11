#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np


def communicate_distribution_function(self):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the distribution
    function array.
    """

    # Accessing the values of the global and local Vectors
    local_value = self._da.getVecArray(self._local)
    glob_value = self._da.getVecArray(self._glob)

    N_ghost = self.N_ghost

    # Global value is non-inclusive of the ghost-zones:
    glob_value[:] = (np.array(self.f))[N_ghost:-N_ghost, N_ghost:-N_ghost, :]

    # The following function takes care of periodic boundary conditions,
    # and interzonal communications:
    self._da.globalToLocal(self._glob, self._local)

    # Converting back from PETSc.Vec to af.Array:
    self.f = af.to_array(local_value[:])

    af.eval(self.f)
    return


def communicate_fields(self):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the EM field
    arrays.
    """

    # Accessing the values of the global and local Vectors
    local_value = self._da_fields.getVecArray(self._local_fields)
    glob_value = self._da_fields.getVecArray(self._glob_fields)

    N_ghost = self.N_ghost

    # Assigning the values of the af.Array fields quantities
    # to the PETSc.Vec:
    (local_value[:])[:, :, 0] = np.array(self.E1)
    (local_value[:])[:, :, 1] = np.array(self.E2)
    (local_value[:])[:, :, 2] = np.array(self.E3)

    (local_value[:])[:, :, 3] = np.array(self.B1)
    (local_value[:])[:, :, 4] = np.array(self.B2)
    (local_value[:])[:, :, 5] = np.array(self.B3)

    # Global value is non-inclusive of the ghost-zones:
    glob_value[:] = (local_value[:])[N_ghost:-N_ghost, N_ghost:-N_ghost, :]

    # Takes care of boundary conditions and interzonal communications:
    self._da_fields.globalToLocal(self._glob_fields, self._local_fields)

    # Converting back to af.Array
    self.E1 = af.to_array((local_value[:])[:, :, 0])
    self.E2 = af.to_array((local_value[:])[:, :, 1])
    self.E3 = af.to_array((local_value[:])[:, :, 2])

    self.B1 = af.to_array((local_value[:])[:, :, 3])
    self.B2 = af.to_array((local_value[:])[:, :, 4])
    self.B3 = af.to_array((local_value[:])[:, :, 5])

    return
