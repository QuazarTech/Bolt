import numpy as np
import arrayfire as af
import N_32
import N_64
import N_128
import N_256
import N_512

af.set_backend("cpu")

from lts.evolve import EM_fields_evolve

final_time = 0.1
time_step  = 0.001
time       = np.arange(0, final_time, time_step)

delta_E_x_hat = 0.01 + 0.02*1j
delta_E_y_hat = 0.05 + 0.01*1j
delta_B_z_hat = 0

data_E_x_LT = np.zeros(time.size)
data_E_y_LT = np.zeros(time.size)
data_B_z_LT = np.zeros(time.size)

for time_index, t0 in enumerate(time):
  delta_B_z_hat, delta_E_x_hat, delta_E_y_hat = EM_fields_evolve(delta_E_x_hat, delta_E_y_hat,\
                                                                 0, 0, 0, delta_B_z_hat, 0, 0,\
                                                                 0, time_step, 2*np.pi, 4*np.pi
                                                                )
  data_E_x_LT[time_index] = np.abs(delta_E_x_hat)
  data_E_y_LT[time_index] = np.abs(delta_E_y_hat)
  data_B_z_LT[time_index] = np.abs(delta_B_z_hat)

# Check for the FDTD solver:

from cks.initialize import set, calculate_x, calculate_y
import cks.fdtd as fdtd

config     = []
config_32  = set(N_32)
config.append(config_32)
config_64  = set(N_64)
config.append(config_64)
config_128 = set(N_128)
config.append(config_128)
config_256 = set(N_256)
config.append(config_256)
config_512 = set(N_512)
config.append(config_512)

error_E_x = np.zeros(len(config))
error_E_y = np.zeros(len(config))
error_B_z = np.zeros(len(config))

import pylab as pl 

for i in range(len(config)):
  X = calculate_x(config[i])[:, :, 0, 0]
  Y = calculate_y(config[i])[:, :, 0, 0]

  E_x = 0.01 * af.cos(config[i].k_x*X + config[i].k_y*Y) - 0.02 * af.sin(config[i].k_x*X + config[i].k_y*Y)
  E_y = 0.05 * af.cos(config[i].k_x*X + config[i].k_y*Y) - 0.01 * af.sin(config[i].k_x*X + config[i].k_y*Y)
  B_z = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)

  data = np.zeros(time.size)

  for time_index, t0 in enumerate(time):
    B_z, E_x, E_y    = fdtd.mode2_fdtd(config[i], B_z, E_x, E_y, 0, 0, time_step)
    data[time_index] = af.max(B_z)

  E_x_LT = delta_E_x_hat.real * af.cos(config[i].k_x*X + config[i].k_y*Y) -\
           delta_E_x_hat.imag * af.sin(config[i].k_x*X + config[i].k_y*Y)

  E_y_LT = delta_E_y_hat.real * af.cos(config[i].k_x*X + config[i].k_y*Y) -\
           delta_E_y_hat.imag * af.sin(config[i].k_x*X + config[i].k_y*Y)

  B_z_LT = delta_B_z_hat.real * af.cos(config[i].k_x*X + config[i].k_y*Y) -\
           delta_B_z_hat.imag * af.sin(config[i].k_x*X + config[i].k_y*Y)

  error_E_x[i] = af.sum(af.abs(E_x[1:-1, 1:-1] - E_x_LT[1:-1, 1:-1]))/E_x[1:-1, 1:-1].elements()
  error_E_y[i] = af.sum(af.abs(E_y[1:-1, 1:-1] - E_y_LT[1:-1, 1:-1]))/E_y[1:-1, 1:-1].elements()
  error_B_z[i] = af.sum(af.abs(B_z[1:-1, 1:-1] - B_z_LT[1:-1, 1:-1]))/B_z[1:-1, 1:-1].elements()

x = np.log10(2**np.arange(5, 10))

poly_E_x = np.polyfit(x, np.log10(error_E_x), 1)
poly_E_y = np.polyfit(x, np.log10(error_E_y), 1)
poly_B_z = np.polyfit(x, np.log10(error_B_z), 1)

print(poly_E_x)
print(poly_E_y)
print(poly_B_z)





