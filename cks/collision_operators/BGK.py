import numpy as np
import arrayfire as af
import cks.compute_moments

def f_MB(da, args):
  # In order to compute the local Maxwell-Boltzmann distribution, the moments of
  # the distribution function need to be computed. For this purpose, all the functions
  # which are passed to the array need to be in velocitiesExpanded form

  config = args.config
  f      = args.f

  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z

  dv_x = (2*config.vel_x_max)/config.N_vel_x
  dv_y = (2*config.vel_y_max)/config.N_vel_y
  dv_z = (2*config.vel_z_max)/config.N_vel_z
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background
  vel_bulk_z_background  = config.vel_bulk_z_background

  # NOTE: Here we are making the assumption that when mode == '2V'/'1V', N_vel_z = 1
  # If otherwise code will break here.
  if(config.mode == '3V'):
    n          = af.tile(cks.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    T          = af.tile(cks.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_x = af.tile(cks.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_y = af.tile(cks.compute_moments.calculate_vel_bulk_y(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_z = af.tile(cks.compute_moments.calculate_vel_bulk_z(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T))**(3/2) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_z - vel_bulk_z)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_z - vel_bulk_z_background)**2/\
                          (2*boltzmann_constant*temperature_background))

  elif(config.mode == '2V'):
    n          = af.tile(cks.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    T          = af.tile(cks.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_x = af.tile(cks.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_y = af.tile(cks.compute_moments.calculate_vel_bulk_y(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                          (2*boltzmann_constant*temperature_background))
  else:
    n          = af.tile(cks.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    T          = af.tile(cks.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_x = af.tile(cks.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], 1
                        )

    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))

  normalization = af.sum(f_background)*dv_x*dv_y*dv_z/(f.shape[0])
  f_MB          = f_MB/normalization

  af.eval(f_MB)
  return(f_MB)

def collision_step_BGK(da, args, dt):

  tau = args.config.tau

  # Converting from positionsExpanded form to velocitiesExpanded form:
  args.f = cks.convert.to_velocitiesExpanded(da, args.config, args.f)

  # Performing the step of df/dt = C[f] = -(f - f_MB)/tau:
  f0             = f_MB(da, args)
  f_intermediate = args.f - (dt/2)*(args.f - f0)/tau
  args.f         = args.f - (dt)  *(f_intermediate - f0)/tau

  # Converting from velocitiesExpanded form to positionsExpanded form:
  args.f = cks.convert.to_positionsExpanded(da, args.config, args.f)

  af.eval(args.f)
  return(args.f)