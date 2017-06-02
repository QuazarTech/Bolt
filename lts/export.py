import numpy as np 
from lts.initialize import f_background

def export_4D_distribution_function(config, delta_f_hat):

  N_x     = config.N_x
  N_vel_x = config.N_vel_x
  k_x     = config.k_x

  N_y     = config.N_y
  N_vel_y = config.N_vel_y
  k_y     = config.k_y

  x_start = config.x_start
  x_end   = config.x_end
  dx      = (x_end - x_start)/N_x

  y_start = config.y_start
  y_end   = config.y_end
  dy      = (y_end - y_start)/N_y

  i = 0.5 + np.arange(0, N_x, 1)
  x = x_start + i * dx
  j = 0.5 + np.arange(0, N_y, 1)
  y = y_start + j * dy

  x, y   = np.meshgrid(x, y)
  f_dist = np.zeros([N_y, N_x, N_vel_y, N_vel_x])
  
  for i in range(N_vel_y):
    for j in range(N_vel_x):
      f_dist[:, :, i, j] = (delta_f_hat[i, j] * np.exp(1j*k_x*x + 1j*k_y*y)).real

  for i in range(N_vel_y):
    for j in range(N_vel_x):
      f_dist[:, :, i, j] += ((f_background(config))[i, j] * np.exp(1j*0*x + 1j*0*y)).real

  return(f_dist)