#!/usr/bin/env python 
# -*- coding: utf-8 -*-

def fields_step(self, dt):

  charge_electron = self.physical_system.params.charge_electron

  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

  p1 = args.vel_x
  p1 = args.vel_y
  vel_z = args.vel_z

  # Convert to velocitiesExpanded:
  args.f = non_linear_solver.convert.to_velocitiesExpanded(da, config, args.f)

  if(config.fields_solver == 'electrostatic'):
    E_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    E_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    E_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    
    B_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    B_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    B_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)

    rho_array = charge_electron * (non_linear_solver.compute_moments.calculate_density(args) - \
                                   config.rho_background
                                  )#(i + 1/2, j + 1/2)
    
    # Passing the values non-inclusive of the ghost zones:
    rho_array = af.moddims(rho_array,\
                           N_y_local + 2 * N_ghost,\
                           N_x_local + 2 * N_ghost
                          )

    # rho_array = np.array(rho_array)[N_ghost:-N_ghost,\
    #                                 N_ghost:-N_ghost
    #                                ]
    
    # E_x, E_y =\
    # solve_electrostatic_fields(da, config, rho_array)

    rho_array = (rho_array)[N_ghost:-N_ghost,\
                                    N_ghost:-N_ghost
                                   ]

    args.E_x[3:-3, 3:-3], args.E_y[3:-3, 3:-3] = fft_poisson(rho_array, config.dx, config.dy)

    args = non_linear_solver.communicate.communicate_fields(da, args, local, glob)

    E_x = args.E_x 
    E_y = args.E_y
  else:
    # Will returned a flattened array containing the values of J_x,y,z in 2D space:
    args.J_x = charge_electron * non_linear_solver.compute_moments.calculate_mom_bulk_x(args) #(i + 1/2, j + 1/2)
    args.J_y = charge_electron * non_linear_solver.compute_moments.calculate_mom_bulk_y(args) #(i + 1/2, j + 1/2)
    args.J_z = charge_electron * non_linear_solver.compute_moments.calculate_mom_bulk_z(args) #(i + 1/2, j + 1/2)

    # We'll convert these back to 2D arrays to be able to perform FDTD:
    args.J_x = af.moddims(args.J_x,\
                          N_y_local + 2 * N_ghost,\
                          N_x_local + 2 * N_ghost
                         )
    
    args.J_y = af.moddims(args.J_y,\
                          N_y_local + 2 * N_ghost,\
                          N_x_local + 2 * N_ghost
                         )

    args.J_z = af.moddims(args.J_z,\
                          N_y_local + 2 * N_ghost,\
                          N_x_local + 2 * N_ghost
                         )

    # Obtaining the values for current density on the Yee-Grid:
    args.J_x = 0.5 * (args.J_x + af.shift(args.J_x, 1, 0)) #(i + 1/2, j)
    args.J_y = 0.5 * (args.J_y + af.shift(args.J_y, 0, 1)) #(i, j + 1/2)
    args.J_z = 0.25 * (args.J_z + af.shift(args.J_z, 1, 0) +\
                       af.shift(args.J_z, 0, 1) + af.shift(args.J_z, 1, 1)
                      ) #(i, j)

    # Storing the values for the previous half-time step:
    # We do this since the B values on the CK grid are defined at time t = n
    # While the B values on the FDTD grid are defined at t = n + 1/2
    B_x_old, B_y_old, B_z_old = args.B_x.copy(), args.B_y.copy(), args.B_z.copy()
    
    args = fdtd(da, args, local, glob, 0.5*dt)

    E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(args.E_x, args.E_y, args.E_z,\
                                                        B_x_old, B_y_old, B_z_old
                                                       )
    args = fdtd(da, args, local, glob, 0.5*dt)

  # Tiling such that E_x, E_y and B_z have the same array dimensions as f:
  # This is required to perform the interpolation in velocity space:
  # NOTE: Here we are making the assumption that when mode == '2V'/'1V', N_vel_z = 1
  # If otherwise code will break here.
  if(config.mode == '3V'):
    E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
 
  else:
    E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

  F_x = charge_electron * (E_x + vel_y * B_z - vel_z * B_y) #(i + 1/2, j + 1/2)
  F_y = charge_electron * (E_y - vel_x * B_z + vel_z * B_x) #(i + 1/2, j + 1/2)
  F_z = charge_electron * (E_z - vel_y * B_x + vel_x * B_y) #(i + 1/2, j + 1/2)

  args.f = f_interp_vel_3d(args, F_x, F_y, F_z, dt)

  # Convert to positionsExpanded:
  args.f = non_linear_solver.convert.to_positionsExpanded(da, args.config, args.f)

  af.eval(args.f)
  return(args)