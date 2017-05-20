import arrayfire as af

def f_interp_2d(args, dt):

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  x      = args.x 
  y      = args.y 

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  
  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary

  length_x = right_boundary - left_boundary
  length_y = top_boundary - bot_boundary

  N_ghost_x = config.N_ghost_x
  N_ghost_y = config.N_ghost_y

  x_new  = x - vel_x*dt
  dx     = af.sum(x[0, 1, 0, 0] - x[0, 0, 0, 0])

  # Imposing periodic boundary conditions on the system:
  # NOTE:CHANGE TO BE MADE HERE WHEN CONSIDERING OTHER B.C's  
 
  while(af.sum(x_new<left_boundary)!=0):
    x_new = af.select(x_new<left_boundary,
                      x_new + length_x,
                      x_new
                     )

  while(af.sum(x_new>right_boundary)!=0):
    x_new = af.select(x_new>right_boundary,
                      x_new - length_x,
                      x_new
                     )

  y_new  = y - vel_y*dt
  dy     = af.sum(y[1, 0, 0, 0] - y[0, 0, 0, 0])  

  # Imposing periodic boundary conditions on the system:
  # NOTE:CHANGE TO BE MADE HERE WHEN CONSIDERING OTHER B.C's 

  while(af.sum(y_new<bot_boundary)!=0):
      y_new = af.select(y_new<bot_boundary,
                        y_new + length_y,
                        y_new
                       )

  while(af.sum(y_new>top_boundary)!=0):
      y_new = af.select(y_new>top_boundary,
                        y_new - length_y,
                        y_new
                       )

  x_interpolant = (x_new - left_boundary)/dx + N_ghost_x
  y_interpolant = (y_new - bot_boundary)/dy + N_ghost_y
  
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
  
  vel_x_max       = config.vel_x_max
  vel_y_max       = config.vel_y_max

  vel_x_new = vel_x - dt * af.tile(F_x, 1, 1,\
                                   f.shape[2],\
                                   f.shape[3]
                                  )
  
  vel_y_new = vel_y - dt * af.tile(F_y, 1, 1,\
                                   f.shape[2],\
                                   f.shape[3]
                                  )

  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])
     
  vel_x_interpolant = (vel_x_new + vel_x_max)/dv_x
  vel_y_interpolant = (vel_y_new + vel_y_max)/dv_y
  
  f_interp = af.approx2(af.reorder(f, 2, 3, 0, 1),\
                        af.reorder(vel_y_interpolant, 2, 3, 0, 1),\
                        af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
                        af.INTERP.BICUBIC_SPLINE
                       )
  
  f_interp = af.reorder(f_interp, 2, 3, 0, 1)

  af.eval(f_interp)
  return(f_interp)