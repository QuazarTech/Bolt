import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
pl.rcParams['figure.dpi']      = 80
pl.rcParams['image.cmap']      = 'gist_heat'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 30
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

dt      = 0.001
t_final = 5.0
time    = np.arange(dt, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2

h5f   = h5py.File('dump/0000.h5', 'r')
r     = h5f['q1'][:].reshape(N_q1, N_q2)
theta = h5f['q2'][:].reshape(N_q1, N_q2)
n0    = h5f['n'][:].reshape(N_q1, N_q2)
h5f.close()

x = r * np.cos(theta)
y = r * np.sin(theta)

pl.contourf(x, y, n0, 100)
pl.title('Time = 0')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.axes().set_aspect('equal')
# pl.xlim([0, 2])
# pl.ylim([-2, 2])
pl.savefig('images/0000.png')
pl.clf()

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    n   = h5f['n'][:].reshape(N_q1, N_q2)
    h5f.close()

    pl.contourf(x, y, n, 100)
    pl.title('Time = %.3f'%t0)
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    # pl.xlim([0, 2])
    # pl.ylim([-2, 2])
    pl.axes().set_aspect('equal')
    pl.savefig('images/%04d'%(time_index + 1) + '.png')
    pl.clf()
