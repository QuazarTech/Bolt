import arrayfire as af

def calculate_density(args):

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x  = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y  = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  density = af.sum(af.sum(f, 3)*dv_x, 2)*dv_y
  
  af.eval(density)
  return(density)
  
def calculate_mom_bulk_x(args):

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  momentum_x = af.sum(af.sum(f * vel_x, 3)*dv_x, 2)*dv_y
  
  af.eval(momentum_x)
  return(momentum_x)

def calculate_vel_bulk_x(args):

  vel_bulk_x = calculate_mom_bulk_x(args)/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_mom_bulk_y(args):

  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])

  momentum_y = af.sum(af.sum(f * vel_y, 3)*dv_x, 2)*dv_y
  
  af.eval(momentum_y)
  return(momentum_y)

def calculate_vel_bulk_y(args):

  vel_bulk_y = calculate_mom_bulk_y(args)/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_temperature(args):

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