import lts.initialize
import lts.evolve

import cks.initialize
import cks.evolve

import pylab as pl
import arrayfire as af
import numpy as np

import params_files.N_32 as N_32
import params_files.N_64 as N_64
import params_files.N_128 as N_128
import params_files.N_256 as N_256
import params_files.N_512 as N_512

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20  
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

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


def test_case():
  config     = []
  config_32  = cks.initialize.set(N_32)
  config.append(config_32)
  config_64  = cks.initialize.set(N_64)
  config.append(config_64)
  config_128 = cks.initialize.set(N_128)
  config.append(config_128)
  config_256 = cks.initialize.set(N_256)
  config.append(config_256)
  config_512 = cks.initialize.set(N_512)
  config.append(config_512)

  print(af.info())

  error = np.zeros(len(config))

  for i in range(len(config)):
    x      = cks.initialize.calculate_x(config[i])
    vel_x  = cks.initialize.calculate_vel_x(config[i])

    f_initial  = cks.initialize.f_initial(config[i])
    time_array = cks.initialize.time_array(config[i])

    class args:
        pass

    args.config = config[i]
    args.f      = f_initial
    args.vel_x  = vel_x
    args.x      = x

    data, f_final = cks.evolve.time_integration(args, time_array)

    delta_f_hat_initial = lts.initialize.init_delta_f_hat(config[i])
    time_array          = lts.initialize.time_array(config[i])

    delta_rho_hat, delta_f_hat_final = lts.evolve.time_integration(config[i],\
                                                                   delta_f_hat_initial,\
                                                                   time_array)

    N_x     = config[i].N_x
    N_vel_x = config[i].N_vel_x
    k_x     = config[i].k_x

    x = np.linspace(0, 1, N_x)
    f_dist = np.zeros([N_x, N_vel_x])
    for j in range(N_vel_x):
      f_dist[:, j] = (delta_f_hat_final[j] * np.exp(1j*k_x*x)).real

    f_background = cks.initialize.f_background(config[i])

    error[i] = af.sum(af.abs(af.to_array(f_dist) + f_background[3:-3] - f_final[3:-3]))/f_dist.size

  x = np.log10(2**np.arange(5, 10))

  poly = np.polyfit(x, np.log10(error), 1)

  assert(abs(poly[0]+2)<0.2)