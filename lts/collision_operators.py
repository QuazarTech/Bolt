import numpy as np
from lts.initialize import f_background

def BGK_collision_operator(config, delta_f_hat):

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  vel_x     = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x      = vel_x[1] - vel_x[0]

  vel_y_max = config.vel_y_max
  N_vel_y   = config.N_vel_y
  vel_y     = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
  dv_y      = vel_y[1] - vel_y[0]

  vel_x, vel_y = np.meshgrid(vel_x, vel_y)
  tau          = config.tau

  normalization = np.sum(f_background(config)) * dv_x * dv_y

  if(config.mode == '2V'):
    delta_rho_hat = np.sum(delta_f_hat) * dv_x * dv_y
    delta_v_x_hat = np.sum(delta_f_hat * vel_x) * dv_x * dv_y/rho_background
    delta_v_y_hat = np.sum(delta_f_hat * vel_y) * dv_x * dv_y/rho_background
    delta_T_hat   = np.sum(delta_f_hat * (0.5*(vel_x**2 + vel_y**2) -\
                                          temperature_background
                                          )) * dv_x * dv_y/rho_background

  
    expr_term_1 = delta_T_hat * mass_particle**2 * rho_background * vel_x**2
    expr_term_2 = delta_T_hat * mass_particle**2 * rho_background * vel_y**2
    expr_term_3 = 2 * temperature_background**2 * delta_rho_hat * boltzmann_constant * mass_particle
    expr_term_4 = 2 * (delta_v_x_hat * mass_particle**2 * rho_background*vel_x +\
                       delta_v_y_hat * mass_particle**2 * rho_background *vel_y -\
                       delta_T_hat * boltzmann_constant * mass_particle *rho_background
                      )*temperature_background
    
    C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/\
            (4*np.pi*boltzmann_constant**2*temperature_background**3)*\
            np.exp(-mass_particle/(2*boltzmann_constant*temperature_background) * \
                  (vel_x**2 + vel_y**2)))/normalization - delta_f_hat)/tau
  
  elif(config.mode == '1V'):
    delta_rho_hat = np.sum(delta_f_hat) * dv_x * dv_y
    delta_v_x_hat = np.sum(delta_f_hat * vel_x) * dv_x * dv_y/rho_background
    delta_T_hat   = np.sum(delta_f_hat * (vel_x**2 - temperature_background)) *\
                    dv_x * dv_y/rho_background
    
    expr_term_1 = np.sqrt(2 * mass_particle**3) * delta_T_hat * rho_background * vel_x**2
    expr_term_2 = 2 * np.sqrt(2 * mass_particle) * boltzmann_constant * delta_rho_hat * \
                  temperature_background**2
    expr_term_3 = 2 * np.sqrt(2 * mass_particle**3) * rho_background * delta_v_x_hat * vel_x * \
                  temperature_background
    expr_term_4 = - np.sqrt(2 * mass_particle) * boltzmann_constant * delta_T_hat *\
                    rho_background * temperature_background
    
    C_f = ((((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)*\
           np.exp(-mass_particle * vel_x**2/(2 * boltzmann_constant * temperature_background))/\
           (4 * np.sqrt(np.pi * temperature_background**5 * boltzmann_constant**3)))/\
            normalization - delta_f_hat
           )/tau
          )

  
  return C_f
