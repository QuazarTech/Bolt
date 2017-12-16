#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

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

    # Assigning the local array only when Dirichlet
    # boundary conditions are applied. This is needed since
    # only inflowing characteristics are to be changed by 
    # the apply boundary conditions function.

    if(   self.boundary_conditions.in_q1_left   == 'dirichlet'
       or self.boundary_conditions.in_q1_right  == 'dirichlet' 
       or self.boundary_conditions.in_q2_bottom == 'dirichlet' 
       or self.boundary_conditions.in_q2_top    == 'dirichlet' 
      ):
        af.flat(self.f).to_ndarray(self._local_f_array)

    # Global value is non-inclusive of the ghost-zones:
    af.flat(self.f[:, N_g:-N_g, N_g:-N_g]).to_ndarray(self._glob_f_array)
    # af.moddims(self.f[:, N_g:-N_g, N_g:-N_g], 
    #            self.f[:, N_g:-N_g, N_g:-N_g].elements()
    #           ).to_ndarray(self._glob_f_array)

    # The following function takes care of interzonal communications
    # Additionally, it also automatically applies periodic BCs when necessary
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


def communicate_fields(self, on_fdtd_grid = False):
    """
    Used in communicating the values at the boundary zones
    for each of the local vectors among all procs.
    This routine is called to take care of communication
    (and periodic B.C's) procedures for the EM field
    arrays. The function may be used for communicating the
    field values at (i, j) which is used by default. Additionally,
    it can also be used to communicate the values on the Yee-grid
    which is used by the FDTD solver.
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
        flattened_global_EM_fields_array = \
            af.flat(self.yee_grid_EM_fields[:, N_g:-N_g, N_g:-N_g])
        flattened_global_EM_fields_array.to_ndarray(self._glob_fields_array)

    else:
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
