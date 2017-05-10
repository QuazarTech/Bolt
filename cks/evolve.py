import numpy as np
import arrayfire as af

from cks.compute_moments import calculate_density, calculate_vel_bulk_x,\
                                calculate_vel_bulk_y, calculate_temperature
                                
def f_MB(args):
  """
  Return the local Maxwell Boltzmann distribution function. This distribution function is 
  Maxwellian while maintaining the bulk parameters of the original distribution function which
  was passed to it.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

  Output:
  -------
    f_MB : Local Maxwell Boltzmann distribution array.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  if(config.mode == '2D2V'):
    vel_y      = args.vel_y
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))  
  
  else:
    n          = af.tile(calculate_density(args), 1, f.shape[1])
    T          = af.tile(calculate_temperature(args), 1, f.shape[1])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1])
    
    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))

  af.eval(f_MB)
  return(f_MB)

def collision_step(args, dt):
  """
  Performs the collision step where df/dt = C[f] is solved for
  In this function the BGK collision operator is being solved for.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_final : Returns the distribution function after performing the collision step
  """
  tau = args.config.tau
  f   = args.f 

  f0             = f_MB(args)
  f_intermediate = f - (dt/2)*(f - f0)/tau
  f_final        = f - (dt)*(f_intermediate - f0)/tau

  af.eval(f_final)
  return(f_final)

def fields_step(args, dt):
  """
  Solves for the field step where df/dt + E df/dv is solved for

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_fields : Returns the distribution function after performing the fields step.
  """
  config = args.config
  f      = args.f
  
  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary
  dx             = length_x/(N_x - 1) 
  
  charge_particle = config.charge_particle


  from cks.poisson_solvers import fft_poisson
  from cks.boundary_conditions.periodic import periodic_x, periodic_y
  from cks.interpolation_routines import f_interp_vel_1d, f_interp_vel_2d
  from cks.fdtd import fdtd, fdtd_grid_to_ck_grid, ck_grid_to_fdtd_grid

  if(config.mode == '2D2V'):

    vel_x  = args.vel_x
    vel_y  = args.vel_y
    
    E_x = args.E_x
    E_y = args.E_y
    E_z = args.E_z

    B_x = args.B_x
    B_y = args.B_y
    B_z = args.B_z

    E_x, E_y, E_z, B_x, B_y, B_z = ck_grid_to_fdtd_grid(config, E_x, E_y, E_z, B_x, B_y, B_z)
    
    J_x = charge_particle * calculate_vel_bulk_x(args) * calculate_density(args)
    J_y = charge_particle * calculate_vel_bulk_y(args) * calculate_density(args)

    J_x = 0.25 * (J_x + af.shift(J_x, 0, -1) + af.shift(J_x, 1, 0) + af.shift(J_x, 1, -1))

    J_x = periodic_x(config, J_x)
    J_x = periodic_y(config, J_x)
     
    E_x, E_y, E_z, B_x_new, B_y_new, B_z_new = fdtd(config, E_x, E_y, E_z, B_x, B_y, B_z,\
                                                    J_x, J_y, 0, dt)

    # To account for half-time steps:
    B_x = 0.5 * (B_x + B_x_new)
    B_y = 0.5 * (B_y + B_y_new)
    B_z = 0.5 * (B_z + B_z_new)

    E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(config, E_x, E_y, E_z, B_x, B_y, B_z)

    F_x      = charge_particle * (E_x + vel_y[:, :, 0, 0] * B_z)
    F_y      = charge_particle * (E_y - vel_x[:, :, 0, 0] * B_z)
    f_fields = f_interp_vel_2d(args, F_x, F_y, dt)

  else:
    E_x       = af.constant(0, f.shape[0], dtype = af.Dtype.c64)
    E_x_local = fft_poisson(charge_particle*
                            calculate_density(args)[N_ghost_x:-N_ghost_x-1],\
                            dx
                           )
    E_x_local = af.join(0, E_x_local, E_x_local[0])
    
    E_x[N_ghost_x:-N_ghost_x] = E_x_local
    E_x                       = periodic_x(config, E_x)     
    
    f_fields = f_interp_vel_1d(args, af.real(E_x), dt)
    
  args.f   = f_fields
  args.B_x = B_x
  args.B_y = B_y
  args.B_z = B_z
  args.E_x = E_x
  args.E_y = E_y
  args.E_z = E_z

  return(args)

def time_integration(args, time_array):
  """
  The main function that evolves the system in time.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f_initial : 2D/4D distribution function that is passed to the function. This is the 
                initial condition for the simulation, and is typically obtained by using 
                the appropriately named function from the initialize sub-module.

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

    x : 2D/4D array that contains the variations in x along the 0th/1st axis

    y : If applicable it is the 4D array that contains the variations in y 
        along the 0th axis
    
    time_array : Array that contains the values of time at which the 
                 simulation evaluates the physical quantities.  

  Output:
  -------
    data : Contains data about the amplitude of density in the system with progression
           in time.

    f_current : This array returned is the distribution function that is obtained
                in the final time step.
  """
  config = args.config
    
  data      = np.zeros(time_array.size)

  from cks.interpolation_routines import f_interp_1d, f_interp_2d
  from cks.boundary_conditions.periodic import periodic_x, periodic_y

  for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0):
        print("Computing for Time = ", t0)

    dt = time_array[1] - time_array[0]
    if(config.mode == '2D2V'):
      args.f = f_interp_2d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)
      args.f = periodic_y(config, args.f)
      args.f = collision_step(args, 0.5*dt)
      args.f = periodic_x(config, args.f)
      args.f = periodic_y(config, args.f)
      args.f = f_interp_2d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)
      args.f = periodic_y(config, args.f)
      args   = fields_step(args, dt)
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
      
    else:
      args.f = f_interp_1d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)
      args.f = collision_step(args, 0.5*dt)
      args.f = periodic_x(config, args.f)
      args.f = f_interp_1d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)
      args   = fields_step(args, dt)
      args.f = periodic_x(config, args.f)
      args.f = f_interp_1d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)
      args.f = collision_step(args, 0.5*dt)
      args.f = periodic_x(config, args.f)
      args.f = f_interp_1d(args, 0.25*dt)
      args.f = periodic_x(config, args.f)

    data[time_index] = af.max(calculate_density(args))

  return(data, args.f)
