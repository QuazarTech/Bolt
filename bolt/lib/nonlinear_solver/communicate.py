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

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

    N_g = self.N_ghost

    af.flat(self.f).to_ndarray(self._local_f_array)
    # Global value is non-inclusive of the ghost-zones:
    af.flat(self.f[:, N_g:-N_g, N_g:-N_g]).to_ndarray(self._glob_f_array)
    
    # The following function takes care of periodic boundary conditions,
    # and interzonal communications:
    self._da_f.globalToLocal(self._glob_f, self._local_f)

    # Converting back from PETSc.Vec to af.Array:
    f_flattened = af.to_array(self._local_f_array)
    self.f      = af.moddims(f_flattened,
                             self.N_p1 * self.N_p2 * self.N_p3 
                             N_q1_local + 2 * N_g
                             N_q2_local + 2 * N_g
                            )

    af.eval(self.f)
    return


def communicate_fields(self, on_fdtd_grid=False):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the EM field
    arrays.
    """

    N_g = self.N_ghost

    # Assigning the values of the af.Array 
    # fields quantities to the PETSc.Vec:

    if(on_fdtd_grid is True):
        joined_E_fields = af.join(0, self.E1_fdtd, self.E2_fdtd, self.E3_fdtd)
        joined_B_fields = af.join(0, self.B1_fdtd, self.B2_fdtd, self.B3_fdtd)

        flattened_EM_fields_array = af.flat(af.join(0, joined_E_fields, joined_B_fields))
        flattened_EM_fields_array.to_ndarray(self._local_fields_array)

    else:
        joined_E_fields = af.join(0, self.E1, self.E2, self.E3)
        joined_B_fields = af.join(0, self.B1, self.B2, self.B3)

        flattened_EM_fields_array = af.flat(af.join(0, joined_E_fields, joined_B_fields))
        flattened_EM_fields_array.to_ndarray(self._local_fields_array)

    # Global value is non-inclusive of the ghost-zones:
    self._glob_fields_array = ((self._local_fields_array).\
                               reshape(6, 
                                       self.E1.shape[1],
                                       self.E1.shape[2]
                                      )[:, N_g:-N_g, N_g:-N_g]).ravel()
    

    # Takes care of boundary conditions and interzonal communications:
    self._da_fields.globalToLocal(self._glob_fields, self._local_fields)

    # Converting back to af.Array
    EM_fields_array = af.moddims(af.to_array(self._local_fields_array),
                                 6, self.E1.shape[1], self.E1.shape[2]
                                )

    if(on_fdtd_grid is True):

        self.E1_fdtd = EM_fields_array[0]
        self.E2_fdtd = EM_fields_array[1]
        self.E3_fdtd = EM_fields_array[2]

        self.B1_fdtd = EM_fields_array[3]
        self.B2_fdtd = EM_fields_array[4]
        self.B3_fdtd = EM_fields_array[5]
        
        af.eval(self.E1_fdtd, self.E2_fdtd, self.E3_fdtd,
                self.B1_fdtd, self.B2_fdtd, self.B3_fdtd
               )

    else:

        self.E1 = EM_fields_array[0]
        self.E2 = EM_fields_array[1]
        self.E3 = EM_fields_array[2]

        self.B1 = EM_fields_array[3]
        self.B2 = EM_fields_array[4]
        self.B3 = EM_fields_array[5]

        af.eval(self.E1, self.E2, self.E3, self.B1, self.B2, self.B3)

    return
