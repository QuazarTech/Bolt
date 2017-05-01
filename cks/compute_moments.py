"""
This module contains the functions which are used in the computation
of the moments of the distribution function such as the density, 
bulk velocities and temperature.
"""
import arrayfire as af

def calculate_density(args):
  """
  This function evaluates and returns the density in the 1D/2D space, 
  depending on the dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along 
            axis 1 in the 2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3
  
  Output:
  -------
    density : The density array is returned by this function after computing the 
              moments.The values in the array indicate the values of density with 
              changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    density = af.sum(af.sum(f, 2)*dv_x, 3)*dv_y
  
  else:
    dv_x    = af.sum(vel_x[0, 1]-vel_x[0, 0])
    density = af.sum(f, 1)*dv_x

  af.eval(density)
  return(density)
  
def calculate_vel_bulk_x(args):
  """
  This function evaluates and returns the x-component of bulk velocity in the 1D/2D space, 
  depending on the dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1 in the
            2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable: it is the 4D velocity array which has the variations in y-velocity 
            along axis 3
  
  Output:
  -------
    vel_bulk_x : The vel_bulk_x array is returned by this function after computing the moments. The 
                 values in the array indicate the values of vel_bulk_x with changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x


  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    momentum_x = af.sum(af.sum(f * vel_x, 2)*dv_x, 3)*dv_y
    vel_bulk_x = momentum_x/calculate_density(args)
  
  else:
    dv_x        = af.sum(vel_x[0, 1]-vel_x[0, 0])
    momentum_x  = af.sum(f*vel_x, 1)*dv_x
    vel_bulk_x  = momentum_x/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_vel_bulk_y(args):
  """
  This function evaluates and returns the y-component of bulk velocity in the 4D space.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along axis 2

    vel_y : 4D velocity array which has the variations in y-velocity along axis 3
  
  Output:
  -------
    vel_bulk_y : The vel_bulk_y array is returned by this function after computing the moments. The 
                 values in the array indicate the values in the y-component of bulk velocity with 
                 changes in x and y.
  """
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

  momentum_y = af.sum(af.sum(f * vel_y, 2)*dv_x, 3)*dv_y
  vel_bulk_y = momentum_y/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_temperature(args):
  """
  This function evaluates and returns the temperature in the 1D/2D space, depending on the
  dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1 in the
            2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable it is the 4D velocity array which has the variations in y-velocity 
            along axis 3
  
  Output:
  -------
    temperature : The temperature array is returned by this function after computing the moments. The 
                  values in the array indicate the values of temperature with changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])

    pressure    = 0.5 * af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + (vel_y-vel_bulk_y)**2), 2)*dv_x, 3)*dv_y
    temperature = pressure/calculate_density(args)
  
  else:
    dv_x        = af.sum(vel_x[0, 1]-vel_x[0, 0])
    vel_bulk_x  = af.tile(calculate_vel_bulk_x(args), 1, vel_x.shape[1])
    temperature = af.sum(f*(vel_x-vel_bulk_x)**2, 1)*dv_x
    temperature = temperature/calculate_density(args)

  af.eval(temperature)
  return(temperature)