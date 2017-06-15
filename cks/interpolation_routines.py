import arrayfire as af

def f_interp_2d(da, args, dt):
  # All the arrays need to be in positionsExpanded form
  config = args.config
  f      = args.f

  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  
  # We need velocity arrays to be in positionExpanded form:
  vel_x = af.moddims(args.vel_x,                   
                     (N_y_local + 2 * N_ghost),\
                     (N_x_local + 2 * N_ghost),\
                     N_vel_y*N_vel_x*N_vel_z,\
                     1
                    )

  vel_y = af.moddims(args.vel_y,                   
                     (N_y_local + 2 * N_ghost),\
                     (N_x_local + 2 * N_ghost),\
                     N_vel_y*N_vel_x*N_vel_z,\
                     1
                    )
  
  x_center = args.x_center 
  y_center = args.y_center 

  x_center_new = x_center - vel_x*dt
  y_center_new = y_center - vel_y*dt

  x_start = config.x_start
  y_start = config.y_start

  dx = af.sum(x_center[0, 1, 0, 0]-x_center[0, 0, 0, 0])
  dy = af.sum(y_center[1, 0, 0, 0]-y_center[0, 0, 0, 0])
  
  # Obtaining the center coordinates:
  (j_center, i_center) = (j_bottom + 0.5, i_left + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  left_boundary = x_start + i_center*dx
  bot_boundary  = y_start + j_center*dy

  # Adding N_ghost to account for the offset due to ghost zones:
  x_interpolant = (x_center_new - left_boundary)/dx + N_ghost
  y_interpolant = (y_center_new - bot_boundary )/dy + N_ghost

  f = af.approx2(f,\
                 y_interpolant,\
                 x_interpolant,\
                 af.INTERP.BICUBIC_SPLINE
                )

  af.eval(f)
  return(f)

def f_interp_vel_3d(args, F_x, F_y, F_z, dt):
  # All the arrays need to be in velocitiesExpanded form
  config = args.config
  f      = args.f
  
  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z
  
  vel_x_max = config.vel_x_max
  vel_y_max = config.vel_y_max
  vel_z_max = config.vel_z_max

  vel_x_new = vel_x - dt * F_x
  vel_y_new = vel_y - dt * F_y
  vel_z_new = vel_z - dt * F_z

  dv_x = (2*vel_x_max)/config.N_vel_x
  dv_y = (2*vel_y_max)/config.N_vel_y
  dv_z = (2*vel_z_max)/config.N_vel_z
  
  # Transforming vel_interpolant to go from [0, N_vel - 1]:
  vel_x_interpolant = (vel_x_new + af.sum(vel_x[0, 0, 0, 0]))/dv_x
  vel_y_interpolant = (vel_y_new + af.sum(vel_y[0, 0, 0, 0]))/dv_y
  vel_z_interpolant = (vel_z_new + af.sum(vel_z[0, 0, 0, 0]))/dv_z
  
  # Reordering to bring the variation in values along axis 0
  f = af.approx1(af.reorder(f),\
                 af.reorder(vel_y_interpolant),\
                 af.INTERP.CUBIC_SPLINE
                )
  
  f = af.reorder(f)
  # print(f.shape)
  # print(vel_x_interpolant.shape)
  # print(vel_z_interpolant.shape)

  f = af.approx2(af.reorder(f, 2, 3, 0, 1),\
                 af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
                 af.reorder(vel_z_interpolant, 2, 3, 0, 1),\
                 af.INTERP.BICUBIC_SPLINE
                )

  # Reordering back to the original convention(velocitiesExpanded):
  f = af.reorder(f, 2, 3, 0, 1)

  af.eval(f)
  return(f)