#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import arrayfire as af
from lib.nonlinear_solver.EM_fields_solver.electrostatic import fft_poisson

def fields_step(self, dt):

  charge_electron = self.physical_system.params.charge_electron

  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

  if(self.physical_system.params.fields_solver == 'electrostatic'):
    self._communicate_fields()

    self.E1 = compute
    self.E2
    
  else:
    # Will returned a flattened array containing the values of J_x,y,z in 2D space:
    self.J1 = self.physical_system.params.charge_electron * self.compute_moments('J1_bulk') #(i + 1/2, j + 1/2)
    self.J2 = self.physical_system.params.charge_electron * self.compute_moments('J2_bulk') #(i + 1/2, j + 1/2)
    self.J3 = self.physical_system.params.charge_electron * self.compute_moments('J3_bulk') #(i + 1/2, j + 1/2)

    # Obtaining the values for current density on the Yee-Grid:
    self.J1 = 0.5 *  (self.J1 + af.shift(self.J2, 0, 1)) #(i + 1/2, j)
    self.J2 = 0.5 *  (self.J2 + af.shift(self.J2, 1, 0)) #(i, j + 1/2)
   
    self.J3 = 0.25 * (args.J_z + af.shift(args.J_z, 1, 0) +\
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
  return