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
    viewer = PETSc.Viewer().createHDF5(file_name, 
                                       PETSc.Viewer.Mode.READ, 
                                       comm=self._comm
                                      )
    self._glob_f.load(viewer)

    N_g = self.N_ghost
    self.f[:, N_g:-N_g, N_g:-N_g] = af.moddims(af.to_array(self._glob_f_array),
                                               self.N_p1 * self.N_p2 * self.N_p3,
                                               self.N_q1_local, self.N_q2_local
                                              )

    return
