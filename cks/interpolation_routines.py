"""
In the Cheng-Knorr method, interpolation steps need to be performed
in the phase space coordinates to solve the system considered. 
For instance,
df/dt + v df/dx = 0 has its solution given by:
f(x, v, t) = f(x - v*t, v, 0)
This module contains the interpolation routines performed in both
the phase space coordinates, namely velocity and position in our
case.
"""

import arrayfire as af

def f_interp_2d(da, args, dt):
  """
  Performs the advection step in position space
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    Object args is also passed to the function of which the following attributes are 
    utilized:

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this file

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2

    x : 4D array that contains the variations in x along the 1st axis

    y : 4D array that contains the variations in y along the 0th axis

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the 
               interpolation step in position space
  """

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
  x_interpolant = (x_new - left_boundary)/dx + 0*N_ghost
  y_interpolant = (y_new - bot_boundary )/dy + 0*N_ghost

  f_interp = af.approx2(f,\
                        y_interpolant,\
                        x_interpolant,\
                        af.INTERP.BICUBIC_SPLINE
                       )

  af.eval(f_interp)
  return(f_interp)

def f_interp_vel_2d(args, F_x, F_y, dt):
  """
  Performs the interpolation in velocity space. This function is
  used in solving for the fields contribution in the Boltzmann equation.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this file

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2

    F_x   : x-component of the EM force 

    F_y   : y-component of the EM force
    
    dt    : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the
               interpolation step in velocity space.

  """
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