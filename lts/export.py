import numpy as np 
import h5py
import lts.initialize as initialize

def export_data(config, density_data, final_delta_f_hat):
  """
  Used to export the data that has been generated in the simulation run.

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

    density_data : Array that contains the values of amplitude of density perturbation
                   at different times. The values in density_data correspond to the time 
                   given by the same index in time_array.

    final_delta_f_hat : Array containing the values of the mode perturbation at
                        the final time step. 

  Output:
  -------
    None : No data is returned by this function. However, a HDF5 file called data.h5 is created
           This file contains the distribution function at the final time, along with the density data
           and its corresponding time_array.
  """
  N_x     = config.N_x
  N_vel_x = config.N_vel_x
  k_x     = config.k_x

  x = np.linspace(0, 1, N_x)
  f_dist = np.zeros([N_x, N_vel_x])
  for i in range(N_vel_x):
    f_dist[:, i] = (final_delta_f_hat[i] * np.exp(1j*k_x*x)).real
    
  # Temporary change to run convergence tests:
  return f_dist
  
  # h5f = h5py.File('data.h5', 'w')
  # h5f.create_dataset('density', data = density_data)
  # h5f.create_dataset('time',    data = initialize.time_array(config))
  # h5f.create_dataset('f_dist',  data = f_dist)
  # h5f.close()