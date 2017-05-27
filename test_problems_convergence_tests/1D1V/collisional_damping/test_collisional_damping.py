# Importing parameter files which will be used in the run.
import params_files.N_32 as N_32
import params_files.N_64 as N_64
import params_files.N_128 as N_128
import params_files.N_256 as N_256
import params_files.N_512 as N_512

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

# Importing solver library functions
import setup
import cks.initialize
import cks.evolve
import lts.initialize
import lts.evolve

import arrayfire as af
import numpy as np

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