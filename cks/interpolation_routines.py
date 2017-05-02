"""
In the Cheng-Knorr method, interpolation steps need to be performed
in the phase space coordinates to solve the system considered. 
For instance,
df/dt + v df/dx = 0 has its solution given by:
f(x, v, t) = f(x - v*t, v, 0)
This module contains the interpolation routines performed in both
the phase space coordinates, namely velocity and position in our
case. The interpolation functions are available for 1D-1V, and
2D-2V systems, and are names appropriately.
"""
import arrayfire as af

def f_interp_1d(args, dt):
  """
  Performs the advection step in 1D + 1V space.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following 
    attributes are utilized:

    config: Object config which is obtained by set() is passed to 
            this file

    f : 2D distribution function that is passed to the function

    vel_x : 2D velocity array which has the variations in x-velocity 
            along axis 1

    x : 2D array that contains the variations in x along the 0th axis

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the 
               interpolation step in configuration space.
  """

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  x      = args.x 

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary

  N_ghost_x = config.N_ghost_x

  x_new     = x - vel_x*dt
  step_size = af.sum(x[1,0] - x[0,0])
  f_interp  = af.constant(0, f.shape[0], f.shape[1], dtype = af.Dtype.f64)
  
  # Interpolating:
  
  x_temp = x_new[N_ghost_x:-N_ghost_x, :]
  
  # Imposing periodic boundary conditions on the system:
  # NOTE:CHANGE TO BE MADE HERE WHEN CONSIDERING OTHER B.C's

  while(af.sum(x_temp<left_boundary)!=0):
    x_temp = af.select(x_temp<left_boundary,
                       x_temp + length_x,
                       x_temp
                      )

  while(af.sum(x_temp>right_boundary)!=0):
    x_temp = af.select(x_temp>right_boundary,
                       x_temp - length_x,
                       x_temp
                      )
  
  x_interpolant = x_temp/step_size + N_ghost_x
  
  f_interp[N_ghost_x:-N_ghost_x, :] = af.approx1(f, x_interpolant,\
                                                 af.INTERP.CUBIC_SPLINE
                                                )
 
  af.eval(f_interp)
  return(f_interp)

def f_interp_2d(args, dt):
  """
  Performs the advection step in 2D + 2V space.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following 
    attributes are utilized:

    config: Object config which is obtained by set() is passed to 
            this file

    f : 4D distribution function that is passed to the function.

    vel_x : 4D velocity array which has the variations in 
            x-velocity along axis 2

    vel_y : 4D velocity array which has the variations in 
            y-velocity along axis 3

    x : 4D array that contains the variations in x along the 1st axis

    y : 4D array that contains the variations in y along the 0th axis

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the 
               2D interpolation step in configuration space
  """
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

  N_x = config.N_x
  N_y = config.N_y

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  f_interp  = af.constant(0,\
                          N_y + 2*N_ghost_y,\
                          N_x + 2*N_ghost_x,\
                          N_vel_x,\
                          N_vel_y,\
                          dtype = af.Dtype.f64
                         )
  
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

  x_interpolant = x_new/dx
  y_interpolant = y_new/dy
  
  f_interp = af.approx2(f,\
                        y_interpolant,\
                        x_interpolant,\
                        af.INTERP.BICUBIC_SPLINE
                       )
  
  af.eval(f_interp)
  return(f_interp)

def f_interp_vel_1d(args, E_x, dt):
  """
  Performs the interpolation in 1V velocity space. This function is
  used in solving for the fields contribution in the Boltzmann equation.

  Parameters:
  -----------
    Object args is passed to the function of which the following 
    attributes are utilized:

    config: Object config which is obtained by set() is passed to 
            this file

    f : 2D distribution function that is passed to the function. 

    vel_x : 2D velocity array which has the variations in x-velocity 
            along axis 1

    E_x : x-component of the electric field. 

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the 
               interpolation step in velocity space.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  charge_particle = config.charge_particle
  vel_x_max       = config.vel_x_max

  v_new     = vel_x - charge_particle * dt * af.tile(E_x, 1, f.shape[1])
  step_size = af.sum(vel_x[0,1] - vel_x[0,0])
  
  # Interpolating:
     
  v_interpolant = (v_new + vel_x_max)/step_size
  
  f_interp      = af.approx1(af.reorder(f),\
                             af.reorder(v_interpolant),\
                             af.INTERP.CUBIC_SPLINE
                            )
  
  f_interp = af.reorder(f_interp)

  af.eval(f_interp)
  return(f_interp)

def f_interp_vel_2d(args, F_x, F_y, dt):
  """
  Performs the interpolation in 2V velocity space. This function is
  used in solving for the fields contribution in the Boltzmann equation.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    config: Object config which is obtained by set() is passed to this file

    f     : 4D distribution function that is passed to the function. 

    vel_x : 4D velocity array which has the variations in x-velocity 
            along axis 2

    vel_y : 4D velocity array which has the variations in y-velocity 
            along axis 3

    E_x   : x-component of the electric field. 

    E_y   : y-component of the electric field.
    
    dt    : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the  2D
               interpolation step in velocity space.

  """
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

  dv_x = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])
     
  vel_x_interpolant = (vel_x_new + vel_x_max)/dv_x
  vel_y_interpolant = (vel_y_new + vel_y_max)/dv_y
  
  f_interp      = af.approx2(af.reorder(f, 2, 3, 0, 1),\
                             af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
                             af.reorder(vel_y_interpolant, 2, 3, 0, 1),\
                             af.INTERP.BICUBIC_SPLINE
                            )
  
  f_interp = af.reorder(f_interp, 2, 3, 0, 1)

  af.eval(f_interp)
  return(f_interp)