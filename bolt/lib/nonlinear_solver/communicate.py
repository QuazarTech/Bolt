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

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

    N_g = self.N_ghost

    # Assigning the local array only when non-periodic 
    # boundary conditions are applied:
    if(   self.boundary_conditions.in_q1 != 'periodic'
       or self.boundary_conditions.in_q2 != 'periodic' 
      ):
        af.flat(self.f).to_ndarray(self._local_f_array)

    # Global value is non-inclusive of the ghost-zones:
    af.flat(self.f[:, N_g:-N_g, N_g:-N_g]).to_ndarray(self._glob_f_array)
    
    # The following function takes care of periodic boundary conditions,
    # and interzonal communications:
    self._da_f.globalToLocal(self._glob_f, self._local_f)

    # Converting back from PETSc.Vec to af.Array:
    f_flattened = af.to_array(self._local_f_array)
    self.f      = af.moddims(f_flattened,
                             self.N_p1 * self.N_p2 * self.N_p3,
                             N_q1_local + 2 * N_g,
                             N_q2_local + 2 * N_g
                            )

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

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

    N_g = self.N_ghost

    # Assigning the values of the af.Array 
    # fields quantities to the PETSc.Vec:

    if(on_fdtd_grid is True):
        if(   self.boundary_conditions.in_q1 != 'periodic'
           or self.boundary_conditions.in_q2 != 'periodic' 
          ):
            flattened_EM_fields_array = af.flat(self.yee_grid_EM_fields)
            flattened_EM_fields_array.to_ndarray(self._local_fields_array)
        
        flattened_global_EM_fields_array = \
            af.flat(self.yee_grid_EM_fields[:, N_g:-N_g, N_g:-N_g])
        flattened_global_EM_fields_array.to_ndarray(self._glob_fields_array)

    else:
        if(   self.boundary_conditions.in_q1 != 'periodic'
           or self.boundary_conditions.in_q2 != 'periodic' 
          ):
            flattened_EM_fields_array = af.flat(self.cell_centered_EM_fields)
            flattened_EM_fields_array.to_ndarray(self._local_fields_array)

        flattened_global_EM_fields_array = \
            af.flat(self.cell_centered_EM_fields[:, N_g:-N_g, N_g:-N_g])
        flattened_global_EM_fields_array.to_ndarray(self._glob_fields_array)

    # Takes care of boundary conditions and interzonal communications:
    self._da_fields.globalToLocal(self._glob_fields, self._local_fields)


    # Converting back to af.Array
    if(on_fdtd_grid is True):

        self.yee_grid_EM_fields = af.moddims(af.to_array(self._local_fields_array),
                                             6, N_q1_local + 2 * N_g,
                                             N_q2_local + 2 * N_g
                                            )
        
        af.eval(self.yee_grid_EM_fields)

    else:

        self.cell_centered_EM_fields = af.moddims(af.to_array(self._local_fields_array),
                                                  6, N_q1_local + 2 * N_g,
                                                  N_q2_local + 2 * N_g
                                                 )
        
        af.eval(self.cell_centered_EM_fields)
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_communicate_fields += toc - tic

    return
