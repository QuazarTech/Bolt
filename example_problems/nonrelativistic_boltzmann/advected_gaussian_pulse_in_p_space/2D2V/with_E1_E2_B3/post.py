import numpy as np
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 20, 10
pl.rcParams['figure.dpi']      = 100
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

dt      = 0.001
t_final = 1.0
time    = np.arange(0, t_final + dt, dt)

h5f = h5py.File('dump/0000.h5', 'r')
p1  = h5f['p1'][:].reshape(128, 128)
p2  = h5f['p2'][:].reshape(128, 128)
f   = h5f['distribution_function'][:].reshape(128, 128)
h5f.close()

h5f = h5py.File('particle_traj.h5', 'r')
sol = h5f['particle_paths'][:]
h5f.close()

maxf = np.max(f) + 0.02
minf = np.min(f) - 0.02

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    p1  = h5f['p1'][:].reshape(128, 128)
    p2  = h5f['p2'][:].reshape(128, 128)
    f   = h5f['distribution_function'][:].reshape(128, 128)
    h5f.close()

    fig = pl.figure()

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_aspect('equal')
    for i in range(200):
        ax1.plot(sol[time_index, i], sol[time_index, i + 200], 'or')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_aspect('equal')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.contourf(p1, p2, f, np.linspace(minf, maxf, 120), cmap='bwr')

    fig.suptitle('Time = %.3f'%(t0))
    pl.savefig('images/' + '%04d'%time_index + '.png')
    pl.close(fig)
    pl.clf()
