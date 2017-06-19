import numpy as np 
from lts.initialize import f_background

def export_5D_distribution_function(config, delta_f_hat):

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
  f_dist             = np.zeros([N_y, N_x, N_vel_y, N_vel_x, N_vel_z])
  
  for i in range(N_vel_y):
    for j in range(N_vel_x):
      for k in range(N_vel_z):
        f_dist[:, :, i, j, k] = (delta_f_hat[i, j, k] * \
                                np.exp(1j*k_x*x_center + 1j*k_y*y_center)).real

  # Adding the background distribution:
  for i in range(N_vel_y):
    for j in range(N_vel_x):
      for k in range(N_vel_z):
        f_dist[:, :, i, j, k] += ((f_background(config))[i, j, k] * \
                                 np.exp(1j*0*x_center + 1j*0*y_center)).real

  return(f_dist)