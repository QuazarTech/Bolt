# Importing parameter files which will be used in the run.
import params_files.N_32 as N_32
import params_files.N_64 as N_64
import params_files.N_128 as N_128
import params_files.N_256 as N_256
import params_files.N_512 as N_512

# Importing solver library functions
import setup_simulation
import lts.initialize
import lts.evolve
import lts.export

import h5py

config     = []
config_32  = setup_simulation.configuration_object(N_32)
config.append(config_32)
config_64  = setup_simulation.configuration_object(N_64)
config.append(config_64)
config_128 = setup_simulation.configuration_object(N_128)
config.append(config_128)
config_256 = setup_simulation.configuration_object(N_256)
config.append(config_256)
config_512 = setup_simulation.configuration_object(N_512)
config.append(config_512)

for i in range(len(config)):
  print() # Insert blank line
  print("Running LT for N =", config[i].N_x)
  print() # Insert blank line
  
  time_array                       = setup_simulation.time_array(config[i])
  delta_f_hat_initial              = lts.initialize.init_delta_f_hat(config[i])
  delta_rho_hat, delta_f_hat_final = lts.evolve.time_integration(config[i], delta_f_hat_initial, time_array)

  f_lt = lts.export.export_4D_distribution_function(config[i], delta_f_hat_final)

  h5f  = h5py.File('distribution_function_data_files/lt/lt_distribution_function_' \
                    + str(config[i].N_x) + '.h5', 'w'
                  )
  h5f.create_dataset('distribution_function', data = f_lt)
  h5f.close()

  h5f  = h5py.File('density_data_files/lt/lt_density_data_' + str(config[i].N_x) + '.h5', 'w')
  h5f.create_dataset('density_data', data = delta_rho_hat)
  h5f.create_dataset('time', data = time_array)
  h5f.close()