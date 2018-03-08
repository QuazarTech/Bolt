#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import arrayfire as af

def load_distribution_function(self, file_name):
    """
    This function is used to load the distribution function from the
    dump file that was created by dump_distribution_function.

    Parameters
    ----------

    file_name : The distribution_function array will be loaded from this
                provided file name.

    Examples
    --------
    
    >> solver.load_distribution_function('distribution_function')
    
    The above statemant will load the distribution function data stored in the file
    distribution_function.h5 into self.f
    """
    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 
                                       PETSc.Viewer.Mode.READ, 
                                       comm=self._comm
                                      )
    self._glob_f.load(viewer)
    N_g = self.N_ghost

    # Reassigning back the distribution function:
    self.f[:, :, N_g:-N_g, N_g:-N_g] = af.moddims(af.to_array(self._glob_f_array),
                                                  self.N_p1 * self.N_p2 * self.N_p3,
                                                  self.N_species, N_q1_local, N_q2_local
                                                 )

    return

def load_EM_fields(self, file_name):
    """
    This function is used to load the EM fields from the
    dump file that was created by dump_EM_fields.

    Parameters
    ----------

    file_name : The EM_fields array will be loaded from this
                provided file name.

    Examples
    --------
    
    >> solver.load_EM_fields('data_EM_fields')
    
    The above statemant will load the EM fields data stored in the file
    data_EM_fields.h5 into self.cell_centered_EM_fields
    """
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 
                                       PETSc.Viewer.Mode.READ, 
                                       comm=self._comm
                                      )
    
    self.fields_solver._glob_fields.load(viewer)

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self.fields_solver._da_fields.getCorners()

    N_g = self.N_ghost
    
    self.fields_solver.yee_grid_EM_fields[:, :, N_g:-N_g, N_g:-N_g] = \
        af.moddims(af.to_array(self.fields_solver._glob_fields_array), 
                   6, 1, N_q1_local, N_q2_local
                  )

    self.fields_solver.yee_grid_to_cell_centered_grid()
    return
