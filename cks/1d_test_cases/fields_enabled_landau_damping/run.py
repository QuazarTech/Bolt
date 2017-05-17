import cks.initialize as initialize
import cks.evolve as evolve
import pylab as pl
import arrayfire as af
import params
import lts.initialize
import lts.evolve

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

config = initialize.set(params)
x      = initialize.calculate_x(config)
vel_x  = initialize.calculate_vel_x(config)

f_initial  = initialize.f_initial(config)
time_array = initialize.time_array(config)

class args:
    pass

args.config = config
args.f      = f_initial
args.vel_x  = vel_x
args.x      = x

dx = af.sum(x[1, 0] - x[0, 0])

from cks.poisson_solvers import fft_poisson
from cks.compute_moments import calculate_density
charge_particle = config.charge_particle

E_x       = af.constant(0, f_initial.shape[0], dtype = af.Dtype.c64)
E_x_local = fft_poisson(charge_particle*
                        calculate_density(args)[3:-4],\
                        dx
                       )
E_x_local = af.join(0, E_x_local, E_x_local[0])

E_x[3:-3] = E_x_local

E_x_fdtd = 0.5 * (E_x + af.shift(E_x, 0, -1))

E_x_fdtd[:3]  = E_x_fdtd[-7:-4]
E_x_fdtd[-3:] = E_x_fdtd[4:7]

args.E_x = E_x_fdtd

data, f_final = evolve.time_integration(args, time_array)

delta_f_hat_initial = lts.initialize.init_delta_f_hat(config)

delta_rho_hat, delta_f_hat_final = lts.evolve.time_integration(config, delta_f_hat_initial, time_array)

pl.semilogy(time_array, data - 1, label = 'CK')
pl.semilogy(time_array, delta_rho_hat, '--', color = 'black', label = 'LT')
pl.xlabel('Time')
pl.ylabel(r'$MAX(\delta \rho(x))$')
pl.legend()
pl.savefig('plot.png')
