#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import arrayfire as af

from bolt.lib.utils.af_petsc_conversion import petsc_local_array_to_af

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

    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 
                                         PETSc.Viewer.Mode.READ, 
                                         comm=self._comm
                                        )
    self._glob_f.load(viewer)
    self._da_f.globalToLocal(self._glob_f, self._local_f)

    self.f = petsc_local_array_to_af(self, 
                                     self.N_p1*self.N_p2*self.N_p3,
                                     self.N_species,
                                     self._local_f_array
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
    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 
                                         PETSc.Viewer.Mode.READ, 
                                         comm=self._comm
                                        )
    
    self.fields_solver._glob_fields.load(viewer)

    self.fields_solver._da_fields.globalToLocal(self.fields_solver._glob_fields, 
                                                self.fields_solver._local_fields
                                               )
    
    self.fields_solver.yee_grid_EM_fields = \
            petsc_local_array_to_af(self, 6, 1, self.fields_solver._local_fields_array)

    self.fields_solver.yee_grid_to_cell_centered_grid()
    return
