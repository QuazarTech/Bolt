#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import arrayfire as af
from bolt.lib.linear.utils.fft_funcs import fft2

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
    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', PETSc.Viewer.Mode.READ)
    self._glob_f.load(viewer)
    self.f_hat =   2 * fft2(af.to_array(self._glob_f_array.reshape(self.N_p1 * self.N_p2 * self.N_p3,
                                                                   self.N_species, self.N_q1, self.N_q2
                                                                  )
                                       )
                           ) / (self.N_q1 * self.N_q2)

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
    """

    viewer = PETSc.Viewer().createHDF5(file_name + '.h5', PETSc.Viewer.Mode.READ)
    self.fields_solver._glob_fields.load(viewer)

    self.fields_solver.fields_hat = \
        2 * fft2(af.to_array(self.fields_solver._glob_fields_array.reshape(6, 1, 
                                                                           self.N_q1, 
                                                                           self.N_q2
                                                                          )
                            )
                ) / (self.N_q1 * self.N_q2)
