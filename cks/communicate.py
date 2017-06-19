# This file contains the functions that are used to take care of the interzonal
# communications when the code is run in parallel across multiple nodes. 
# Additionally, these functions are also responsible for applying boundary conditions.

import numpy as np
import arrayfire as af

def communicate_distribution_function(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Storing values of af.Array in PETSc.Vec:
  local_value[:] = np.array(args.f)
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # The following function takes care of the boundary conditions, 
  # and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back from PETSc.Vec to af.Array:
  f_updated = af.to_array(local_value[:])

  af.eval(f_updated)
  return(f_updated)

def communicate_fields(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Assigning the values of the af.Array fields quantities
  # to the PETSc.Vec:
  (local_value[:])[:, :, 0] = np.array(args.E_x)
  (local_value[:])[:, :, 1] = np.array(args.E_y)
  (local_value[:])[:, :, 2] = np.array(args.E_z)
  
  (local_value[:])[:, :, 3] = np.array(args.B_x)
  (local_value[:])[:, :, 4] = np.array(args.B_y)
  (local_value[:])[:, :, 5] = np.array(args.B_z)

  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # Takes care of boundary conditions and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back to af.Array
  args.E_x = af.to_array((local_value[:])[:, :, 0])
  args.E_y = af.to_array((local_value[:])[:, :, 1])
  args.E_z = af.to_array((local_value[:])[:, :, 2])

  args.B_x = af.to_array((local_value[:])[:, :, 3])
  args.B_y = af.to_array((local_value[:])[:, :, 4])
  args.B_z = af.to_array((local_value[:])[:, :, 5])

  return(args)