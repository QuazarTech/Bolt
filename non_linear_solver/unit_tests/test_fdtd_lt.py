import numpy as np
import arrayfire as af
import cks.initialize as initialize
import cks.fdtd as fdtd
from lts.evolve import EM_fields_mode1, EM_fields_mode2

import params_files.N_32 as N_32
import params_files.N_64 as N_64
import params_files.N_128 as N_128
import params_files.N_256 as N_256
import params_files.N_512 as N_512

def test_mode1():
  config     = []
  config_32  = initialize.set(N_32)
  config.append(config_32)
  config_64  = initialize.set(N_64)
  config.append(config_64)
  config_128 = initialize.set(N_128)
  config.append(config_128)
  config_256 = initialize.set(N_256)
  config.append(config_256)
  config_512 = initialize.set(N_512)
  config.append(config_512)

  error_B_x = np.zeros(len(config))
  error_B_y = np.zeros(len(config))
  error_E_z = np.zeros(len(config))
  for i in range(len(config)):
    X = initialize.calculate_x(config[i])[:, :, 0, 0]
    Y = initialize.calculate_y(config[i])[:, :, 0, 0]

    left_boundary  = config[i].left_boundary
    right_boundary = config[i].right_boundary
    bot_boundary   = config[i].bot_boundary
    top_boundary   = config[i].top_boundary

    length_x = right_boundary - left_boundary
    length_y = top_boundary - bot_boundary

    N_x = config[i].N_x
    N_y = config[i].N_y

    N_ghost_x = config[i].N_ghost_x
    N_ghost_y = config[i].N_ghost_y

    dx = (length_x/(N_x - 1))
    dy = (length_y/(N_y - 1))

    dt = 0.01 * (32/config[i].N_x) 

    final_time = 0.1
    time       = np.arange(dt, final_time + dt, dt)

    B_x = 0.01 * af.cos(config[i].k_x*X + config[i].k_y*(Y + dy/2)) -\
          0.02 * af.sin(config[i].k_x*X + config[i].k_y*(Y + dy/2))

    B_y = 0.05 * af.cos(config[i].k_x*(X + dx/2) + config[i].k_y*Y) -\
          0.01 * af.sin(config[i].k_x*(X + dx/2) + config[i].k_y*Y)
    
    E_z = af.constant(0, B_x.shape[0], B_x.shape[1], dtype = af.Dtype.f64)

    delta_B_x_hat = 0.01 + 0.02*1j
    delta_B_y_hat = 0.05 + 0.01*1j
    delta_E_z_hat = 0
    delta_J_z_hat = 0

    J_z = 0

    for time_index, t0 in enumerate(time):
      E_z, B_x, B_y = fdtd.mode1_fdtd(config[i], E_z, B_x, B_y, J_z, dt)
      delta_E_z_hat, delta_B_x_hat, delta_B_y_hat = EM_fields_mode1(delta_E_z_hat, delta_B_x_hat,\
                                                                    delta_B_y_hat, delta_J_z_hat,\
                                                                    config[i].k_x, config[i].k_y, dt
                                                                   )

    Bx_lt = delta_B_x_hat.real * af.cos(config[i].k_x*X +\
                                        config[i].k_y*(Y + dy/2)) \
            - delta_B_x_hat.imag * af.sin(config[i].k_x*X +\
                                          config[i].k_y*(Y + dy/2))

    By_lt = delta_B_y_hat.real * af.cos(config[i].k_x*(X + dx/2) +\
                                        config[i].k_y*Y) \
            - delta_B_y_hat.imag * af.sin(config[i].k_x*(X + dx/2) +\
                                          config[i].k_y*Y)

    Ez_lt = delta_E_z_hat.real * af.cos(config[i].k_x*X +\
                                        config[i].k_y*Y) \
            - delta_E_z_hat.imag * af.sin(config[i].k_x*X +\
                                          config[i].k_y*Y)

    error_B_x[i] = af.sum(af.abs(B_x[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Bx_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_B_y[i] = af.sum(af.abs(B_y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 By_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_E_z[i] = af.sum(af.abs(E_z[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ez_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

  x = np.log10(2**np.arange(5, 10))

  poly_B_x = np.polyfit(x, np.log10(error_B_x), 1)
  poly_B_y = np.polyfit(x, np.log10(error_B_y), 1)
  poly_E_z = np.polyfit(x, np.log10(error_E_z), 1)
  
  assert(abs(poly_B_x[0]+2)<0.3 and\
         abs(poly_B_y[0]+2)<0.3 and\
         abs(poly_E_z[0]+2)<0.3
        )

def test_mode2():
  config     = []
  config_32  = initialize.set(N_32)
  config.append(config_32)
  config_64  = initialize.set(N_64)
  config.append(config_64)
  config_128 = initialize.set(N_128)
  config.append(config_128)
  config_256 = initialize.set(N_256)
  config.append(config_256)
  config_512 = initialize.set(N_512)
  config.append(config_512)

  error_E_x = np.zeros(len(config))
  error_E_y = np.zeros(len(config))
  error_B_z = np.zeros(len(config))

  for i in range(len(config)):
    X = initialize.calculate_x(config[i])[:, :, 0, 0]
    Y = initialize.calculate_y(config[i])[:, :, 0, 0]

    left_boundary  = config[i].left_boundary
    right_boundary = config[i].right_boundary
    bot_boundary   = config[i].bot_boundary
    top_boundary   = config[i].top_boundary

    length_x = right_boundary - left_boundary
    length_y = top_boundary - bot_boundary

    N_x = config[i].N_x
    N_y = config[i].N_y

    N_ghost_x = config[i].N_ghost_x
    N_ghost_y = config[i].N_ghost_y

    dx = (length_x/(N_x - 1))
    dy = (length_y/(N_y - 1))

    dt = 0.01 * (32/config[i].N_x) 

    final_time = 0.1
    time       = np.arange(dt, final_time + dt, dt)

    E_x = 0.01 * af.cos(config[i].k_x*(X + dx/2) + config[i].k_y*Y) -\
          0.02 * af.sin(config[i].k_x*(X + dx/2) + config[i].k_y*Y)
    E_y = 0.05 * af.cos(config[i].k_x*X + config[i].k_y*(Y + dy/2)) -\
          0.01 * af.sin(config[i].k_x*X + config[i].k_y*(Y + dy/2))

    B_z = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)

    delta_E_x_hat = 0.01 + 0.02*1j
    delta_E_y_hat = 0.05 + 0.01*1j
    delta_B_z_hat = 0
    delta_J_x_hat = 0
    delta_J_y_hat = 0

    J_x, J_y = 0, 0

    for time_index, t0 in enumerate(time):
      B_z, E_x, E_y = fdtd.mode2_fdtd(config[i], B_z, E_x, E_y, J_x, J_y, dt)
      delta_B_z_hat, delta_E_x_hat, delta_E_y_hat = EM_fields_mode2(delta_B_z_hat, delta_E_x_hat,\
                                                                    delta_E_y_hat, delta_J_x_hat,\
                                                                    delta_J_y_hat, config[i].k_x,\
                                                                    config[i].k_y, dt
                                                                   )


    Ex_lt = delta_E_x_hat.real * af.cos(config[i].k_x*(X + dx/2) +\
                                        config[i].k_y*Y
                                       ) \
            - delta_E_x_hat.imag * af.sin(config[i].k_x*(X + dx/2) +\
                                          config[i].k_y*Y
                                         )

    Ey_lt = delta_E_y_hat.real * af.cos(config[i].k_x*X +\
                                        config[i].k_y*(Y + dy/2)
                                       ) \
             - delta_E_y_hat.imag * af.sin(config[i].k_x*X +\
                                           config[i].k_y*(Y + dy/2)
                                          )


    Bz_lt = delta_B_z_hat.real * af.cos(config[i].k_x*(X + dx/2) +\
                                        config[i].k_y*(Y + dy/2)
                                       ) \
             - delta_B_z_hat.imag * af.sin(config[i].k_x*(X + dx/2) +\
                                           config[i].k_y*(Y + dy/2)
                                          )

    error_E_x[i] = af.sum(af.abs(E_x[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ex_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_E_y[i] = af.sum(af.abs(E_y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ey_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_B_z[i] = af.sum(af.abs(B_z[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Bz_lt[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

  x = np.log10(2**np.arange(5, 10))

  poly_E_x = np.polyfit(x, np.log10(error_E_x), 1)
  poly_E_y = np.polyfit(x, np.log10(error_E_y), 1)
  poly_B_z = np.polyfit(x, np.log10(error_B_z), 1)
  
  assert(abs(poly_E_x[0]+2)<0.3 and\
         abs(poly_E_y[0]+2)<0.3 and\
         abs(poly_B_z[0]+2)<0.3
        )