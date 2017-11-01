import numpy as np
import time

def BGK(self):
    m = 1
    k = 1

    rho = 1 
    T   = 1
  
    tau = self.physical_system.params.tau

    delta_rho_hat = self.compute_moments('density')
    delta_v_x_hat = self.compute_moments('mom_p1_bulk')
    delta_v_y_hat = self.compute_moments('mom_p2_bulk')
    delta_v_z_hat = self.compute_moments('mom_p3_bulk')
    delta_T_hat   = self.compute_moments('energy') #- self.delta_f_hat *self.dp1 * self.dp2 * self.dp3

    print(delta_T_hat)
    time.sleep(5)  
    print()
  # if(config.mode == '3V'):

  #   expr_term_1 = 2 * np.sqrt(2) * T * delta_v_x_hat * m**(5/2) * rho * vel_x
  #   expr_term_2 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_x**2
  #   expr_term_3 = 2 * np.sqrt(2) * T * delta_v_y_hat * m**2.5 * rho * vel_y
  #   expr_term_4 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_y**2
  #   expr_term_5 = 2 * np.sqrt(2) * T * delta_v_z_hat * m**2.5 * rho * vel_z
  #   expr_term_6 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * vel_z**2
  #   expr_term_7 = 2 * np.sqrt(2) * T**2 * delta_rho_hat * k * m**(3/2)
  #   expr_term_8 = (2 * np.sqrt(2) * T - 3 * np.sqrt(2) * delta_T_hat) * T * k * rho * m**(3/2)
  #   expr_term_9 = -2 * np.sqrt(2) * rho * k * T**2 * m**(3/2)

  #   C_f = ((((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4 +\
  #             expr_term_5 + expr_term_6 + expr_term_7 + expr_term_8 + expr_term_9
  #           )*np.exp(-m/(2*k*T) * (vel_x**2 + vel_y**2 + vel_z**2)))/\
  #             (8 * np.pi**1.5 * T**3.5 * k**2.5 * normalization)
  #          ) - delta_f_hat)/tau
  
  # elif(config.mode == '2V'):
  
  #   expr_term_1 = delta_T_hat * m**2 * rho * vel_x**2 
  #   expr_term_2 = delta_T_hat * m**2 * rho * vel_y**2
  #   expr_term_3 = 2 * T**2 * delta_rho_hat * k * m
  #   expr_term_4 = 2 * (delta_v_x_hat * m**2 * rho*vel_x +\
  #                      delta_v_y_hat * m**2 * rho*vel_y -\
  #                      delta_T_hat * k * m *rho
  #                     )*T
    
  #   C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/(4*np.pi*k**2*T**3) * \
  #            np.exp(-m/(2*k*T) * (vel_x**2 + vel_y**2)))/normalization - delta_f_hat)/tau
  
    expr_term_1 = np.sqrt(2 * m**3) * delta_T_hat * rho * self.p1**2
    expr_term_2 = 2 * np.sqrt(2 * m) * k * delta_rho_hat * T**2
    expr_term_3 = 2 * np.sqrt(2 * m**3) * rho * delta_v_x_hat * self.p1 * T
    expr_term_4 = - np.sqrt(2 * m) * k * delta_T_hat * rho * T

    C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4) * \
             np.exp(-m * self.p1**2/(2 * k * T))/(4 * np.sqrt(np.pi * T**5 * k**3))) - \
             self.delta_f_hat)/tau
  
    return C_f