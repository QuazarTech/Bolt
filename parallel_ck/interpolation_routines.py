import arrayfire as af
import numpy as np 

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
  
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()
  i = i_bottom_left + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  j = j_bottom_left + np.arange(-N_ghost, N_y_local + N_ghost, 1)

  left_boundary = x_start + i[N_ghost]*dx
  bot_boundary  = y_start + j[N_ghost]*dy

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