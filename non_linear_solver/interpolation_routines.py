# Since we intend to use this code for a 2D3V simulation run, some manipulations
# need to be performed to allow us to run the same by making use of 4D array structures
# For this purpose, we define 2 forms for every array involved in the calculation:
# positionsExpanded  form : (Ny, Nx, Nvy*Nvx*Nvz, 1)
# velocitiesExpanded form : (Ny*Nx, Nvy, Nvx, Nvz, 1)

import arrayfire as af
import numpy as np
import non_linear_solver.convert
from scipy.interpolate import InterpolatedUnivariateSpline

def f_interp_2d(da, args, dt):
  # Since the interpolation function are being performed in position space,
  # the arrays used in the computation need to be in positionsExpanded form.

  config  = args.config
  N_ghost = config.N_ghost

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  
  # We need velocity arrays to be in positionsExpanded form:
  vel_x = non_linear_solver.convert.to_positionsExpanded(da, config, args.vel_x)
  vel_y = non_linear_solver.convert.to_positionsExpanded(da, config, args.vel_y)

  x_center = args.x_center 
  y_center = args.y_center 

  x_center_new = x_center - vel_x*dt
  y_center_new = y_center - vel_y*dt

  # Obtaining the center coordinates:
  (j_center, i_center) = (j_bottom + 0.5, i_left + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  left_boundary = config.x_start + i_center*config.dx
  bot_boundary  = config.y_start + j_center*config.dy

  # Adding N_ghost to account for the offset due to ghost zones:
  x_interpolant = (x_center_new - left_boundary)/config.dx + N_ghost
  y_interpolant = (y_center_new - bot_boundary )/config.dy + N_ghost

  args.log_f = af.approx2(args.log_f,\
                          y_interpolant,\
                          x_interpolant,\
                          af.INTERP.BICUBIC_SPLINE
                         )

  af.eval(args.log_f)
  return(args.log_f)

def f_interp_vel_3d(args, F_x, F_y, F_z, dt):
  # Since the interpolation function are being performed in velocity space,
  # the arrays used in the computation need to be in velocitiesExpanded form.
  config = args.config
  
  # args.vel_x,y,z are already in velocitiesExpanded form
  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z
  
  # F_x,y,z need to be passed in velocitiesExpanded form
  vel_x_new = vel_x - dt * F_x
  vel_y_new = vel_y - dt * F_y
  vel_z_new = vel_z - dt * F_z

  # Transforming vel_interpolant to go from [0, N_vel - 1]:
  # vel_x_interpolant = (vel_x_new - af.sum(vel_x[0, 0, 0, 0]))/config.dv_x
  # vel_y_interpolant = (vel_y_new - af.sum(vel_y[0, 0, 0, 0]))/config.dv_y
  # vel_z_interpolant = (vel_z_new - af.sum(vel_z[0, 0, 0, 0]))/config.dv_z

  for i in range(vel_x.shape[0]):
    args.log_f[i, 0, :, 0] = af.to_array((InterpolatedUnivariateSpline(np.array(vel_x[i, 0, :, 0]), np.array(args.log_f[i, 0, :, 0]), k = 5))(np.array(vel_x_new[i, 0, :, 0])))
  # We perform the 3d interpolation by performing individual 1d + 2d interpolations:
  # Reordering to bring the variation in values along axis 0 and axis 1

  # Reordering from f(Ny*Nx, vel_y, vel_x, vel_z)     --> f(vel_y, Ny*Nx, vel_x, vel_z)
  # Reordering from vel_y(Ny*Nx, vel_y, vel_x, vel_z) --> vel_y(vel_y, Ny*Nx, vel_x, vel_z)
  # args.log_f = af.approx1(af.reorder(args.log_f),\
  #                         af.reorder(vel_y_interpolant),\
  #                         af.INTERP.CUBIC_SPLINE,\
  #                         off_grid = -46
  #                        )

  # Reordering from f(vel_y, Ny*Nx, vel_x, vel_z)     --> f(vel_x, vel_z, Ny*Nx, vel_y)
  # Reordering from vel_x(Ny*Nx, vel_y, vel_x, vel_z) --> vel_x(vel_x, vel_z, Ny*Nx, vel_y)
  # Reordering from vel_z(Ny*Nx, vel_y, vel_x, vel_z) --> vel_z(vel_x, vel_z, Ny*Nx, vel_y)
  # args.log_f = af.approx2(af.reorder(args.log_f, 2, 3, 1, 0),\
  #                         af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
  #                         af.reorder(vel_z_interpolant, 2, 3, 0, 1),\
  #                         af.INTERP.BICUBIC_SPLINE,\
  #                         off_grid = -46
  #                        )

  # Reordering back to the original convention(velocitiesExpanded):
  # Reordering from f(vel_x, vel_z, Ny*Nx, vel_y) --> f(Ny*Nx, vel_y, vel_x, vel_z)
  # args.log_f = af.reorder(args.log_f, 2, 3, 0, 1)

  af.eval(args.log_f)
  return(args.log_f)