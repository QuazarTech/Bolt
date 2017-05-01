import numpy as np
import arrayfire as af
import cks.initialize as initialize
import cks.fdtd as fdtd
from lts.evolve import EM_fields_mode1, EM_fields_mode2

import N_32
import N_64
import N_128
import N_256
import N_512

import pylab as pl

af.set_backend("cpu")

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

    dt = (dx/4) + (dy/4)

    final_time = 1
    time       = np.arange(0, final_time, dt)

    Ez = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)
    Bx = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)
    By = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)

    Bx[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] =\
    0.01 * af.cos(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
                  config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]) -\
    0.02 * af.sin(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
                  config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x])

    By[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] =\
    0.02 * af.cos(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
                  config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]) -\
    0.04 * af.sin(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
                  config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x])


    delta_B_x_hat = 0.01 + 0.02*1j
    delta_B_y_hat = 0.02 + 0.04*1j
    delta_E_z_hat = 0
    delta_J_z_hat = 0

    Jz = 0

    data = np.zeros(time.size)
    data_lt = np.zeros(time.size)

    for time_index, t0 in enumerate(time):
      Ez, Bx, By = fdtd.mode1_fdtd(config[i], Ez, Bx, By, Jz, dt)
      delta_E_z_hat, delta_B_x_hat, delta_B_y_hat = EM_fields_mode1(delta_E_z_hat, delta_B_x_hat,\
                                                                    delta_B_y_hat, delta_J_z_hat,\
                                                                    config[i].k_x, config[i].k_x, dt
                                                                   )
      data[time_index] = af.max(af.abs(Bx))
      data_lt[time_index] = np.abs(delta_B_x_hat)

    pl.plot(time, data_lt, '--', color = 'black')
    pl.plot(time, data)
    pl.show()


  #   Bx_lt = delta_B_x_hat.real * af.cos(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                       config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]) \
  #            - delta_B_x_hat.imag * af.sin(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                          config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x])

  #   By_lt = delta_B_y_hat.real * af.cos(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                       config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]) \
  #            - delta_B_y_hat.imag * af.sin(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                          config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x])

  #   Ez_lt = delta_E_z_hat.real * af.cos(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                       config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]) \
  #            - delta_E_z_hat.imag * af.sin(config[i].k_x*X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] +\
  #                                          config[i].k_y*Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x])

  #   error_B_x[i] = af.sum(af.abs(Bx[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
  #                                Bx_lt))/\
  #                                (X.shape[0] * X.shape[1])

  #   error_B_y[i] = af.sum(af.abs(By[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
  #                                By_lt))/\
  #                                (X.shape[0] * X.shape[1])

  #   error_E_z[i] = af.sum(af.abs(Ez[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
  #                                Ez_lt))/\
  #                                (X.shape[0] * X.shape[1])

  # x = np.log10(2**np.arange(5, 10))

  # poly_B_x = np.polyfit(x, np.log10(error_B_x), 1)
  # poly_B_y = np.polyfit(x, np.log10(error_B_y), 1)
  # poly_E_z = np.polyfit(x, np.log10(error_E_z), 1)
  
  # print(poly_B_x)
  # assert(abs(poly_B_x[0]+3)<0.3 and\
  #        abs(poly_B_y[0]+3)<0.3 and\
  #        abs(poly_E_z[0]+2)<0.3
  #       )

test_mode1()