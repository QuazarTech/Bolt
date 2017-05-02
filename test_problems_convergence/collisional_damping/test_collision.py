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

print(af.info())

config = cks.initialize.set(N_32)
x      = cks.initialize.calculate_x(config)
vel_x  = cks.initialize.calculate_vel_x(config)

f_initial  = cks.initialize.f_initial(config)
time_array = cks.initialize.time_array(config)

class args:
    pass

args.config = config
args.f      = f_initial
args.vel_x  = vel_x
args.x      = x

data, f_final = cks.evolve.time_integration(args, time_array)

print()

config = lts.initialize.set(N_32)

delta_f_hat_initial = lts.initialize.init_delta_f_hat(config)
time_array          = lts.initialize.time_array(config)

delta_rho_hat, delta_f_hat_final = lts.evolve.time_integration(config, delta_f_hat_initial, time_array)

N_x     = config.N_x
N_vel_x = config.N_vel_x
k_x     = config.k_x

x = np.linspace(0, 1, N_x)
f_dist = np.zeros([N_x, N_vel_x])
for i in range(N_vel_x):
  f_dist[:, i] = (delta_f_hat_final[i] * np.exp(1j*k_x*x)).real

error = af.abs(af.to_array(abs(f_dist)) - f_final[3:-3])

pl.plot(time_array, data - 1)
pl.plot(time_array, delta_rho_hat, '--', color = 'black')
pl.xlabel('Time')
pl.ylabel(r'$MAX(\delta \rho(x))$')
pl.savefig('plot.png')