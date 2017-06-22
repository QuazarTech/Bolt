import numpy as np 
from lts.initialize import f_background

def export_5D_distribution_function(config, delta_f_hat):
  """
  Used to create the 5D distribution function array from the 3V delta_f_hat
  array. This will be used in comparison with the solution as given by the
  Cheng Knorr method.
  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

    delta_f_hat : Array containing the values of the mode perturbation at
                  the final time step. 
  Output:
  -------
    f_dist : 
  """  
  N_x     = config.N_x
  k_x     = config.k_x

  N_y     = config.N_y
  k_y     = config.k_y

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  x_start = config.x_start
  x_end   = config.x_end
  dx      = (x_end - x_start)/N_x

  y_start = config.y_start
  y_end   = config.y_end
  dy      = (y_end - y_start)/N_y

  i_center = 0.5 + np.arange(0, N_x, 1)
  x_center = x_start + i_center * dx
  j_center = 0.5 + np.arange(0, N_y, 1)
  y_center = y_start + j_center * dy

  x_center, y_center = np.meshgrid(x_center, y_center)
  x_center, y_center = x_center.reshape([N_y, N_x, 1, 1, 1]), y_center.reshape([N_y, N_x, 1, 1, 1])
  f_dist             = np.zeros([N_y, N_x, N_vel_y, N_vel_x, N_vel_z])
  
  # Converting delta_f_hat --> delta_f
  f_dist = (delta_f_hat.reshape([1, 1, N_vel_y, N_vel_x, N_vel_z]) * \
            np.exp(1j*k_x*x_center + 1j*k_y*y_center)).real

  # Adding back the background distribution(delta_f --> delta_f + f_background):
  f_dist += np.tile((f_background(config)).reshape(1, 1, N_vel_y, N_vel_x, N_vel_z), (N_y, N_x, 1, 1, 1)) 
  
  return(f_dist)