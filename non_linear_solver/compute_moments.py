# Since we intend to use this code for a 2D3V simulation run, some manipulations
# need to be performed to allow us to run the same by making use of 4D array structures
# For this purpose, we define 2 forms for every array involved in the calculation:
# positionsExpanded  form : (Ny, Nx, Nvy*Nvx*Nvz, 1)
# velocitiesExpanded form : (Ny*Nx, Nvy, Nvx, Nvz, 1)

# Since all the moments need to be computed by performing integrations in velocity
# space, all functions need to have the arrays used specified in velocitiesExpanded form 

import arrayfire as af

def calculate_density(args):
  config = args.config
  f      = af.exp(args.f)

  # n = \int f dv^3
  density = af.sum(af.sum(af.sum(f, 3)*config.dv_z, 2)*config.dv_x, 1)*config.dv_y
  
  af.eval(density)
  return(density)
  
def calculate_mom_bulk_x(args):
  config = args.config
  f      = af.exp(args.f)
  vel_x  = args.vel_x

  # p_x = n v_bulk_x = \int f v_x dv^3
  momentum_x = af.sum(af.sum(af.sum(f*vel_x, 3)*config.dv_z, 2)*config.dv_x, 1)*config.dv_y
  
  af.eval(momentum_x)
  return(momentum_x)

def calculate_vel_bulk_x(args):
  # v_bulk_x = p_x/n
  vel_bulk_x = calculate_mom_bulk_x(args)/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_mom_bulk_y(args):
  config = args.config
  f      = af.exp(args.f)
  vel_y  = args.vel_y

  # p_y = n v_bulk_y = \int f v_y dv^3
  momentum_y = af.sum(af.sum(af.sum(f*vel_y, 3)*config.dv_z, 2)*config.dv_x, 1)*config.dv_y
  
  af.eval(momentum_y)
  return(momentum_y)

def calculate_vel_bulk_y(args):
  # v_bulk_y = p_y/n
  vel_bulk_y = calculate_mom_bulk_y(args)/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_mom_bulk_z(args):
  config = args.config
  f      = af.exp(args.f)
  vel_z  = args.vel_z

  # p_z = n v_bulk_z = \int f v_z dv^3
  momentum_z = af.sum(af.sum(af.sum(f*vel_z, 3)*config.dv_z, 2)*config.dv_x, 1)*config.dv_y
  
  af.eval(momentum_z)
  return(momentum_z)

def calculate_vel_bulk_z(args):
  # v_bulk_z = p_z/n
  vel_bulk_z = calculate_mom_bulk_z(args)/calculate_density(args)
  
  af.eval(vel_bulk_z)
  return(vel_bulk_z)

def calculate_temperature(args):
  config = args.config
  f      = af.exp(args.f)

  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z

  dv_x = config.dv_x
  dv_y = config.dv_y
  dv_z = config.dv_z

  # This condition checking is performed due to this:
  # consider an array declared as a = af.randu(10, 1, 10, 1)
  # a.shape only displays (10, 1, 10). 
  # a.shape[3] will display an out of bounds error
  
  # Tiling the arrays to get them to velocitiesExpanded form:
  if(args.config.N_vel_z != 1):
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
    # 3P = \int ((v_x-vb_x)^2 + (v_y-vb_y)^2 + (v_z - vb_z)^2) f dv^3 
    pressure = (1/3) * af.sum(af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + \
                                               (vel_y-vel_bulk_y)**2 + \
                                               (vel_z-vel_bulk_z)**2
                                              ), 3)*dv_z, 2)*dv_x, 1)*dv_y

  elif(config.mode == '2V'):
    # 2P = \int ((v_x-vb_x)^2 + (v_y-vb_y)^2) f dv^3 
    pressure = 0.5 * af.sum(af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + \
                                             (vel_y-vel_bulk_y)**2
                                            ), 3)*dv_z, 2)*dv_x, 1)*dv_y

  elif(config.mode == '1V'):
    # P = \int ((v_x-vb_x)^2) f dv^3 
    pressure = af.sum(af.sum(af.sum(f*(vel_x-vel_bulk_x)**2, 3)*dv_z, 2)*dv_x, 1)*dv_y
  
  # T = P/n     
  temperature = pressure/calculate_density(args)
    
  af.eval(temperature)
  return(temperature)