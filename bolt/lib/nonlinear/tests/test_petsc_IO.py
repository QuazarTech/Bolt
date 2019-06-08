#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np

N_p1 = 64
N_p2 = 64
N_p3 = 64
N_q1 = 96
N_q2 = 64

N_ghost = 3

da_dump_f = PETSc.DMDA().create([N_q1, N_q2], # Spatial resolution
                                dof = (N_p1*N_p2*N_p3), # number of variables
                                stencil_width = N_ghost
                               )

glob_f    = da_dump_f.createGlobalVec() # PETSc vec
PETSc.Object.setName(glob_f, 'distribution_function')

# Get a numpy array from the PETSc vec
glob_f_array = glob_f.getArray()

# Work with the numpy array. For example,
# time_step(glob_f_array)

# Now dump the data. First create a PETSc Binary MPIIO viewer.
viewer = PETSc.Viewer().createBinary('binary_using_petsc.bin', 'w', comm=PETSc.COMM_WORLD)

# Finally dump. Note that glob_f_array is linked to glob_f.
viewer(glob_f)

# Now try with a PETSc HDF5 viewer.
viewer = PETSc.Viewer().createHDF5('hdf5_using_petsc.h5', 'w', comm=PETSc.COMM_WORLD)
viewer(glob_f)

