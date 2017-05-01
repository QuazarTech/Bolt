import numpy as np
import arrayfire as af
import matplotlib.pyplot as pl
import cks.initialize as initialize
import cks.fdtd as fdtd

import N_32
import N_64
import N_128
import N_256
import N_512

af.set_backend("cpu")

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = False
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['savefig.dpi']     = 100

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

def gauss1D(x, spread):
  return af.exp(-((x - 0.5)**2 )/(2*spread**2))

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
    gauss1D(Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x], 0.1)

    By[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] =\
    gauss1D(X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x], 0.1)

    Ez_initial = Ez.copy()
    Bx_initial = Bx.copy()
    By_initial = By.copy()

    Jz = 0

    for time_index, t0 in enumerate(time):
      Ez, Bx, By = fdtd.mode1_fdtd(config[i], Ez, Bx, By, Jz, dt)

    error_B_x[i] = af.sum(af.abs(Bx[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Bx_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_B_y[i] = af.sum(af.abs(By[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 By_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_E_z[i] = af.sum(af.abs(Ez[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ez_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

  x = np.log10(2**np.arange(5, 10))

  poly_B_x = np.polyfit(x, np.log10(error_B_x), 1)
  poly_B_y = np.polyfit(x, np.log10(error_B_y), 1)
  poly_E_z = np.polyfit(x, np.log10(error_E_z), 1)
  assert(abs(poly_B_x[0]+3)<0.3 and\
         abs(poly_B_y[0]+3)<0.3 and\
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

    dt = (dx/4) + (dy/4)

    final_time = 1
    time       = np.arange(0, final_time, dt)

    Ex = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)
    Ey = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)
    Bz = af.data.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.f64)

    Ex[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] =\
    gauss1D(Y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x], 0.1)

    Ey[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] =\
    gauss1D(X[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x], 0.1)

    Ex_initial = Ex.copy()
    Ey_initial = Ey.copy()
    Bz_initial = Bz.copy()

    Jx, Jy = 0, 0

    for time_index, t0 in enumerate(time):
      Bz, Ex, Ey = fdtd.mode2_fdtd(config[i], Bz, Ex, Ey, Jx, Jy, dt)

    error_E_x[i] = af.sum(af.abs(Ex[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ex_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_E_y[i] = af.sum(af.abs(Ey[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Ey_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

    error_B_z[i] = af.sum(af.abs(Bz[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] -\
                                 Bz_initial[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x]))/\
                                 (X.shape[0] * X.shape[1])

  x = np.log10(2**np.arange(5, 10))

  poly_E_x = np.polyfit(x, np.log10(error_E_x), 1)
  poly_E_y = np.polyfit(x, np.log10(error_E_y), 1)
  poly_B_z = np.polyfit(x, np.log10(error_B_z), 1)

  assert(abs(poly_E_x[0]+3)<0.3 and\
         abs(poly_E_y[0]+3)<0.3 and\
         abs(poly_B_z[0]+2)<0.3
        )