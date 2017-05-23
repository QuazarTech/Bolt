import numpy as np
import arrayfire as af

from cks.compute_moments import calculate_density

def communicate(da, args, local, glob):

  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost
  N_vel_x = args.config.N_vel_x
  N_vel_y = args.config.N_vel_y

  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  local_value[:] = np.array(af.moddims(args.f,\
                                       N_y_local + 2*N_ghost, \
                                       N_x_local + 2*N_ghost, \
                                       N_vel_x * N_vel_y, \
                                       1
                                       )
                            )
  
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  da.globalToLocal(glob, local)

  f_updated = af.moddims(af.to_array(local_value[:]),\
                         N_y_local + 2*N_ghost, \
                         N_x_local + 2*N_ghost, \
                         N_vel_y, \
                         N_vel_x
                        )

  return(f_updated)


def time_integration(da, args, time_array):

    
  data = np.zeros(time_array.size)

  glob  = da.createGlobalVec()
  local = da.createLocalVec()


  from cks.interpolation_routines import f_interp_2d
  
  for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0):
        print("Computing for Time = ", t0)

    dt = time_array[1] - time_array[0]

    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate(da, args, local, glob)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    # args.f = collision_step(args, 0.5*dt)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate(da, args, local, glob)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    # args   = fields_step(args, dt)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate(da, args, local, glob)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    # args.f = collision_step(args, 0.5*dt)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate(da, args, local, glob)
    # args.f = periodic_x(config, args.f)
    # args.f = periodic_y(config, args.f)
      
    data[time_index] = af.max(calculate_density(args))

  return(data, args.f)
