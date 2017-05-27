import arrayfire as af

def f_interp_2d(da, args, dt):

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  x      = args.x 
  y      = args.y 

  x_new  = x - vel_x*dt
  y_new  = y - vel_y*dt

  x_start = config.x_start
  y_start = config.y_start
  N_ghost = config.N_ghost

  dx = af.sum(x[0, 1, 0, 0]-x[0, 0, 0, 0])
  dy = af.sum(y[1, 0, 0, 0]-y[0, 0, 0, 0])
  
  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Obtaining the left, and bottom boundaries for the local zones:
  left_boundary = x_start + i_bottom_left*dx
  bot_boundary  = y_start + j_bottom_left*dy

  # Adding N_ghost to account for the offset due to ghost zones:
  x_interpolant = (x_new - left_boundary)/dx + N_ghost
  y_interpolant = (y_new - bot_boundary )/dy + N_ghost

  f_interp = af.approx2(f,\
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
  
  f_interp = af.reorder(f_interp, 2, 3, 0, 1)

  af.eval(f_interp)
  return(f_interp)