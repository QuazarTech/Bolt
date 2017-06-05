import arrayfire as af

def f_interp_2d(da, args, dt):

  config   = args.config
  f        = args.f
  vel_x    = args.vel_x
  vel_y    = args.vel_y
  x_center = args.x_center 
  y_center = args.y_center 

  x_center_new  = x_center - vel_x*dt
  y_center_new  = y_center - vel_y*dt

  x_start = config.x_start
  y_start = config.y_start
  N_ghost = config.N_ghost

  dx = af.sum(x_center[0, 1, 0, 0]-x_center[0, 0, 0, 0])
  dy = af.sum(y_center[1, 0, 0, 0]-y_center[0, 0, 0, 0])
  
  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the center coordinates:
  (j_center, i_center) = (j_bottom + 0.5, i_left + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  left_boundary = x_start + i_center*dx
  bot_boundary  = y_start + j_center*dy

  # Adding N_ghost to account for the offset due to ghost zones:
  x_interpolant = (x_center_new[N_ghost:-N_ghost, N_ghost:-N_ghost] - left_boundary)/dx + N_ghost
  y_interpolant = (y_center_new[N_ghost:-N_ghost, N_ghost:-N_ghost] - bot_boundary )/dy + N_ghost

  f_interp = af.constant(0, f.shape[0],f.shape[1],f.shape[2],f.shape[3], dtype = af.Dtype.f64)
  
  f_interp[N_ghost:-N_ghost, N_ghost:-N_ghost] = af.approx2(f,\
                                                            y_interpolant,\
                                                            x_interpolant,\
                                                            af.INTERP.BICUBIC_SPLINE
                                                           )

  af.eval(f_interp)
  return(f_interp)

def f_interp_vel_2d(args, F_x, F_y, dt):

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  vel_x_max = config.vel_x_max
  vel_y_max = config.vel_y_max

  vel_x_new = vel_x - dt * F_x
  vel_y_new = vel_y - dt * F_y

  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])
  
  # Transforming vel_interpolant to go from [0, N_vel - 1]:
  vel_x_interpolant = (vel_x_new + vel_x_max)/dv_x
  vel_y_interpolant = (vel_y_new + vel_y_max)/dv_y
  
  # Reordering such that the axis of interpolation is made to be axis 0 and 1
  f_interp = af.approx2(af.reorder(f, 2, 3, 0, 1),\
                        af.reorder(vel_y_interpolant, 2, 3, 0, 1),\
                        af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
                        af.INTERP.BICUBIC_SPLINE
                       )
  
  # Reordering back to the original convention chosen:
  f_interp = af.reorder(f_interp, 2, 3, 0, 1)

  af.eval(f_interp)
  
  return(f_interp)