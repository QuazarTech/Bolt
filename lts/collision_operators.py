import numpy as np
import lts.compute_moments
import lts.initialize

def BGK_collision_operator(config, delta_f_hat):
  """
  Returns the array that contains the values of the linearized BGK collision operator.
  The expression that has been used may be understood more clearly by referring to the
  Sage worksheet on https://goo.gl/dXarsP
  Parameters:
  -----------
    config      : Object config which is obtained by setup_simulation() is 
                  passed to this function
    delta_f_hat : The array of delta_f_hat which is obtained from each step
                  of the time integration. 
  Output:
  -------
    C_f : Array which contains the values of the linearized collision operator. 
  """

  m = config.mass_particle
  k = config.boltzmann_constant

  rho = config.rho_background
  T   = config.temperature_background
  
  vel_x, vel_y, vel_z = config.vel_x, config.vel_y, config.vel_z

  tau           = config.tau

  # Obtaining the normalization constant:
  normalization = lts.initialize.f_background(config, 1)

  delta_rho_hat = lts.compute_moments.delta_rho_hat(config, delta_f_hat)
  delta_v_x_hat = lts.compute_moments.delta_vel_bulk_x_hat(config, delta_f_hat)
  delta_v_y_hat = lts.compute_moments.delta_vel_bulk_y_hat(config, delta_f_hat)
  delta_v_z_hat = lts.compute_moments.delta_vel_bulk_z_hat(config, delta_f_hat)
  delta_T_hat   = lts.compute_moments.delta_T_hat(config, delta_f_hat)
  
  if(config.mode == '3V'):

    expr_term_1 = 2 * np.sqrt(2) * T * delta_v_x_hat * m**(5/2) * rho * vel_x
    expr_term_2 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_x**2
    expr_term_3 = 2 * np.sqrt(2) * T * delta_v_y_hat * m**2.5 * rho * vel_y
    expr_term_4 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_y**2
    expr_term_5 = 2 * np.sqrt(2) * T * delta_v_z_hat * m**2.5 * rho * vel_z
    expr_term_6 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_z**2
    expr_term_7 = 2 * np.sqrt(2) * T**2 * delta_rho_hat * k * m**(3/2)
    expr_term_8 = (2 * np.sqrt(2) * T - 3 * np.sqrt(2) * delta_T_hat) * T * k * rho * m**(3/2)
    expr_term_9 = -2 * np.sqrt(2) * rho * k * T**2 * m**(3/2)

    C_f = ((((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4 +\
              expr_term_5 + expr_term_6 + expr_term_7 + expr_term_8 + expr_term_9
            )*np.exp(-m/(2*k*T) * (vel_x**2 + vel_y**2 + vel_z**2)))/\
              (8 * np.pi**1.5 * T**3.5 * k**2.5 * normalization)
           ) - delta_f_hat)/tau
  
  elif(config.mode == '2V'):
  
    expr_term_1 = delta_T_hat * m**2 * rho * vel_x**2 
    expr_term_2 = delta_T_hat * m**2 * rho * vel_y**2
    expr_term_3 = 2 * T**2 * delta_rho_hat * k * m
    expr_term_4 = 2 * (delta_v_x_hat * m**2 * rho*vel_x +\
                       delta_v_y_hat * m**2 * rho*vel_y -\
                       delta_T_hat * k * m *rho
                      )*T
    
    C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/(4*np.pi*k**2*T**3) * \
             np.exp(-m/(2*k*T) * (vel_x**2 + vel_y**2)))/normalization - delta_f_hat)/tau
  
  elif(config.mode == '1V'):
    
    expr_term_1 = np.sqrt(2 * m**3) * delta_T_hat * rho * vel_x**2
    expr_term_2 = 2 * np.sqrt(2 * m) * k * delta_rho_hat * T**2
    expr_term_3 = 2 * np.sqrt(2 * m**3) * rho * delta_v_x_hat * vel_x * T
    expr_term_4 = - np.sqrt(2 * m) * k * delta_T_hat * rho * T
    
    C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4) * \
             np.exp(-m * vel_x**2/(2 * k * T))/(4 * np.sqrt(np.pi * T**5 * k**3)))/normalization - \
             delta_f_hat)/tau
  
  return C_f