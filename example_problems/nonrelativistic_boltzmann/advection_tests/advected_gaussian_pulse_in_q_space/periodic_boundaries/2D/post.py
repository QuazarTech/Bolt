import numpy as np
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'gist_heat'
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

dt      = 0.0025
t_final = 1.0
time    = np.arange(dt, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2

h5f = h5py.File('dump/0000.h5', 'r')
q1  = h5f['q1'][:].reshape(N_q1, N_q2)
q2  = h5f['q2'][:].reshape(N_q1, N_q2)
n   = h5f['n'][:].reshape(N_q1, N_q2)
h5f.close()

pl.contourf(q1, q2, n, 100)
pl.title('Time = 0')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.axes().set_aspect('equal')
pl.savefig('images/0000.png')
pl.clf()

n_to_plot = 0

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    n   = h5f['n'][:].reshape(N_q1, N_q2)
    h5f.close()

    if((time_index + 1)%40 == 0):
        n_to_plot += n

pl.contourf(q1, q2, n_to_plot, 100)
pl.title('Periodic Boundary Conditions')
# pl.title('Time =' + str(t0))
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.axes().set_aspect('equal')
        # pl.savefig('images/%04d'%((time_index+1)/4) + '.png')
        # pl.clf()

# pl.savefig('trial.svg')
pl.savefig('trial.png')
