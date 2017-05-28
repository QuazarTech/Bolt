"""
This module contains the functions which are used in the computation
of the moments of the distribution function such as the density, 
bulk velocities, momentum and temperature.
"""

import arrayfire as af

def calculate_density(args):
  """
  This function evaluates and returns the density

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2
  
  Output:
  -------
    density : The density array is returned by this function after computing the 
              moments.The values in the array represent the values of density with 
              changes in x and y.
  """

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x  = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y  = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  density = af.sum(af.sum(f, 3)*dv_x, 2)*dv_y
  
  af.eval(density)
  return(density)
  
def calculate_mom_bulk_x(args):
  """
  This function evaluates and returns the bulk momentum in the x-direction

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2
  
  Output:
  -------
    momentum_x : The momentum_x array is returned by this function after computing 
                 the moments.The values in the array represent the values of x-momentum 
                 with changes in x and y.
  """

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  momentum_x = af.sum(af.sum(f * vel_x, 3)*dv_x, 2)*dv_y
  
  af.eval(momentum_x)
  return(momentum_x)

def calculate_vel_bulk_x(args):
  """
  This function evaluates and returns the bulk velocity in the x-direction

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2
  
  Output:
  -------
    vel_bulk_x : The vel_bulk_x array is returned by this function after computing 
                 the moments.The values in the array represent the values of bulk 
                 velocity in the x-direction with changes in x and y.
  """

  vel_bulk_x = calculate_mom_bulk_x(args)/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_mom_bulk_y(args):
  """
  This function evaluates and returns the bulk momentum in the y-direction

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2
  
  Output:
  -------
    momentum_y : The momentum_y array is returned by this function after computing 
                 the moments.The values in the array represent the values of y-momentum 
                 with changes in x and y.
  """

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  momentum_y = af.sum(af.sum(f * vel_y, 3)*dv_x, 2)*dv_y
  
  af.eval(momentum_y)
  return(momentum_y)

def calculate_vel_bulk_y(args):
  """
  This function evaluates and returns the bulk velocity in the y-direction

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes 
    are utilized:

    f : 4D distribution function that is passed to the function. Moments 
        will be computed defined by the state of the system which is indicated 
        by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along 
            axis 3

    vel_y : 4D velocity array which has the variations in y-velocity along
            axis 2
  
  Output:
  -------
    vel_bulk_y : The vel_bulk_y array is returned by this function after computing 
                 the moments.The values in the array represent the values of bulk 
                 velocity in the y-direction with changes in x and y.
  """

  vel_bulk_y = calculate_mom_bulk_y(args)/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_temperature(args):
  """
  This function evaluates and returns the temperature

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
  
  Output:
  -------
    temperature : The temperature array is returned by this function after computing 
                  the moments.The values in the array represent the values of temperature
                  with changes in x and y.
  """

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x  = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y  = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
  vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])

  # The temperature is calculated depending upon the defined dimensionality in
  # velocity space:
  if(config.mode == '1V'):
    pressure = af.sum(af.sum(f*(vel_x-vel_bulk_x)**2, 3)*dv_x, 2)*dv_y
    temperature = pressure/calculate_density(args)
    
  elif(config.mode == '2V'):
    pressure = 0.5 * af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + \
                                      (vel_y-vel_bulk_y)**2
                                     ), 3)*dv_x, 2)*dv_y
        
    temperature = pressure/calculate_density(args)
    
  af.eval(temperature)
  return(temperature)