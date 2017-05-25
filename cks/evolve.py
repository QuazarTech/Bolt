import numpy as np
import arrayfire as af
import cks.initialize as initialize

from cks.compute_moments import calculate_density, calculate_vel_bulk_x,\
                                calculate_vel_bulk_y, calculate_mom_bulk_x,\
                                calculate_mom_bulk_y, calculate_temperature
                                
def f_MB(da, args):

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  if(config.mode == '2V'):
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))  
  
  else:
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))


  normalization = af.sum(initialize.f_background(da, config))*dv_x*dv_y/(vel_x.shape[0] * vel_x.shape[1])
  f_MB          = f_MB/normalization

  af.eval(f_MB)
  return(f_MB)

def collision_step(da, args, dt):

  tau = args.config.tau
  f   = args.f 

  f0             = f_MB(da, args)
  f_intermediate = f - (dt/2)*(f - f0)/tau
  f_final        = f - (dt)*(f_intermediate - f0)/tau

  af.eval(f_final)
  return(f_final)

def fields_step(args, dt):
  
  config = args.config
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  charge_particle = config.charge_particle

  from cks.boundary_conditions.periodic import periodic_x, periodic_y
  from cks.interpolation_routines import f_interp_vel_2d
  from cks.fdtd import fdtd, fdtd_grid_to_ck_grid

  E_x = args.E_x
  E_y = args.E_y
  E_z = args.E_z

  B_x = args.B_x
  B_y = args.B_y
  B_z = args.B_z

  J_x = charge_particle * calculate_mom_bulk_x(args)
  J_y = charge_particle * calculate_mom_bulk_y(args) 
  J_z = af.constant(0, J_x.shape[0], J_x.shape[1])

  J_x = 0.5 * (J_x + af.shift(J_x, 0, -1))
  J_y = 0.5 * (J_y + af.shift(J_y, -1, 0))

  J_x = periodic_x(config, J_x)
  J_x = periodic_y(config, J_x)
  J_y = periodic_x(config, J_y)
  J_y = periodic_y(config, J_y)
   
  E_x, E_y, E_z, B_x_new, B_y_new, B_z_new = fdtd(config,\
                                                  E_x, E_y, E_z,\
                                                  B_x, B_y, B_z,\
                                                  J_x, J_y, J_z,\
                                                  dt
                                                 )

  args.B_x = B_x_new
  args.B_y = B_y_new
  args.B_z = B_z_new
  args.E_x = E_x
  args.E_y = E_y
  args.E_z = E_z

  # To account for half-time steps:
  B_x = 0.5 * (B_x + B_x_new)
  B_y = 0.5 * (B_y + B_y_new)
  B_z = 0.5 * (B_z + B_z_new)

  E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(config, E_x, E_y, E_z, B_x, B_y, B_z)

  F_x      = charge_particle * (E_x) #+ vel_y[:, :, 0, 0] * B_z)
  F_y      = charge_particle * (E_y) #- vel_x[:, :, 0, 0] * B_z)

  f_fields = f_interp_vel_2d(args, F_x, F_y, dt)
    
  args.f   = f_fields

  return(args)

def time_integration(args, time_array):

  config = args.config
    
  data      = np.zeros(time_array.size)

  from cks.interpolation_routines import f_interp_2d
  from cks.boundary_conditions.periodic import periodic_x, periodic_y
  
  for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0):
        print("Computing for Time = ", t0)

    dt = time_array[1] - time_array[0]

    args.f = f_interp_2d(args, 0.25*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    args.f = collision_step(args, 0.5*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(args, 0.25*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    # args   = fields_step(args, dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(args, 0.25*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    args.f = collision_step(args, 0.5*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
    args.f = f_interp_2d(args, 0.25*dt)
    args.f = periodic_x(config, args.f)
    args.f = periodic_y(config, args.f)
      
    data[time_index] = af.max(calculate_density(args))
  return(data, args.f)
