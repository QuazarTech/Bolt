# In this file, we'll need to specify all the arrays in the velocitiesExpanded form

import arrayfire as af

def calculate_density(args):
  f     = args.f

  dv_x  = (2*args.config.vel_x_max)/args.config.N_vel_x
  dv_y  = (2*args.config.vel_y_max)/args.config.N_vel_y
  dv_z  = (2*args.config.vel_z_max)/args.config.N_vel_z

  density = af.sum(af.sum(af.sum(f, 3)*dv_z, 2)*dv_x, 1)*dv_y
  
  af.eval(density)
  return(density)
  
def calculate_mom_bulk_x(args):
  f     = args.f
  vel_x = args.vel_x

  dv_x = (2*args.config.vel_x_max)/args.config.N_vel_x
  dv_y = (2*args.config.vel_y_max)/args.config.N_vel_y
  dv_z = (2*args.config.vel_z_max)/args.config.N_vel_z

  momentum_x = af.sum(af.sum(af.sum(f*vel_x, 3)*dv_z, 2)*dv_x, 1)*dv_y
  
  af.eval(momentum_x)
  return(momentum_x)

def calculate_vel_bulk_x(args):
  vel_bulk_x = calculate_mom_bulk_x(args)/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_mom_bulk_y(args):
  f     = args.f
  vel_y = args.vel_y

  dv_x = (2*args.config.vel_x_max)/args.config.N_vel_x
  dv_y = (2*args.config.vel_y_max)/args.config.N_vel_y
  dv_z = (2*args.config.vel_z_max)/args.config.N_vel_z

  momentum_y = af.sum(af.sum(af.sum(f*vel_y, 3)*dv_z, 2)*dv_x, 1)*dv_y
  
  af.eval(momentum_y)
  return(momentum_y)

def calculate_vel_bulk_y(args):
  vel_bulk_y = calculate_mom_bulk_y(args)/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_mom_bulk_z(args):
  f     = args.f
  vel_z = args.vel_z

  dv_x = (2*args.config.vel_x_max)/args.config.N_vel_x
  dv_y = (2*args.config.vel_y_max)/args.config.N_vel_y
  dv_z = (2*args.config.vel_z_max)/args.config.N_vel_z

  momentum_z = af.sum(af.sum(af.sum(f*vel_z, 3)*dv_z, 2)*dv_x, 1)*dv_y
  
  af.eval(momentum_z)
  return(momentum_z)

def calculate_vel_bulk_z(args):
  vel_bulk_z = calculate_mom_bulk_z(args)/calculate_density(args)
  
  af.eval(vel_bulk_z)
  return(vel_bulk_z)

def calculate_temperature(args):
  config = args.config
  f      = args.f

  vel_x  = args.vel_x
  vel_y  = args.vel_y
  vel_z  = args.vel_z

  dv_x = (2*args.config.vel_x_max)/args.config.N_vel_x
  dv_y = (2*args.config.vel_y_max)/args.config.N_vel_y
  dv_z = (2*args.config.vel_z_max)/args.config.N_vel_z

  if(config.mode == '3V'):
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1], f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, f.shape[1], f.shape[2], f.shape[3])
    vel_bulk_z = af.tile(calculate_vel_bulk_z(args), 1, f.shape[1], f.shape[2], f.shape[3])

  else:
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1], f.shape[2], 1)
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, f.shape[1], f.shape[2], 1)
    vel_bulk_z = af.tile(calculate_vel_bulk_z(args), 1, f.shape[1], f.shape[2], 1)

  # The temperature is calculated depending upon the defined dimensionality in
  # velocity space:
  if(config.mode == '3V'):
    pressure = (1/3) * af.sum(af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + \
                                               (vel_y-vel_bulk_y)**2 + \
                                               (vel_z-vel_bulk_z)**2
                                              ), 3)*dv_z, 2)*dv_x, 1)*dv_y

  elif(config.mode == '2V'):
    pressure = 0.5 * af.sum(af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + \
                                             (vel_y-vel_bulk_y)**2
                                            ), 3)*dv_z, 2)*dv_x, 1)*dv_y

  elif(config.mode == '1V'):
    pressure = af.sum(af.sum(af.sum(f*(vel_x-vel_bulk_x)**2, 3)*dv_z, 2)*dv_x, 1)*dv_y
        
  temperature = pressure/calculate_density(args)
    
  af.eval(temperature)
  return(temperature)