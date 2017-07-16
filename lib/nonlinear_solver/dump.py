#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np 
import h5py
from petsc4py import PETSc

def dump_variables(self, file_name, **args):
  h5f = h5py.File(file_name + '.h5', 'w')
  for i in range(int(len(args)/2)):
    h5f.create_dataset(args[i + int(len(args)/2)], data = args[i])
  h5f.close()
  return

def dump_distribution_function(self, file_name):
  PETSc.Object.setName(self._glob, 'distribution_function')
  viewer = PETSc.Viewer().createHDF5(file_name + '.h5', 'w', comm = self._comm)
  
  global_vec_value    = self._da.getVecArray(self._glob)
  global_vec_value[:] = np.array(self.f[self.N_ghost:-self.N_ghost,\
                                        self.N_ghost:-self.N_ghost, :]
                                )
  viewer(self._glob)
  return