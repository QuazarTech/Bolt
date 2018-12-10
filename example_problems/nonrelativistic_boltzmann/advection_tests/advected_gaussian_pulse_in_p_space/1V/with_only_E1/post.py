import numpy as np
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
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
t_final = 0.4
time    = np.arange(dt, t_final + dt, dt)

h5f = h5py.File('dump/0000.h5', 'r')
p1  = h5f['p1'][:]
f   = h5f['distribution_function'][:]
f_a = h5f['analytic_solution'][:]
h5f.close()

pl.plot(p1[:, 0], f[:, 0], label = 'Species-1 Numerical')
pl.plot(p1[:, 0], f_a[:, 0], '--', color = 'black', label = 'Species-1 Analytic')

pl.plot(p1[:, 1], f[:, 1], label = 'Species-2 Numerical')
pl.plot(p1[:, 1], f_a[:, 1], '-.', color = 'black', label = 'Species-2 Analytic')

pl.legend()
pl.ylim([-1, 1])
pl.title('Time = 0')
pl.xlabel(r'$v$')
pl.ylabel(r'$f$')
pl.savefig('images/0000.png')
pl.clf()

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    p1  = h5f['p1'][:]
    f   = h5f['distribution_function'][:]
    f_a = h5f['analytic_solution'][:]
    h5f.close()

    if((time_index+1)%4==0):

        pl.plot(p1[:, 0], f[:, 0], label = 'Species-1 Numerical')
        pl.plot(p1[:, 0], f_a[:, 0], '--', color = 'black', label = 'Species-1 Analytic')

        pl.plot(p1[:, 1], f[:, 1], label = 'Species-2 Numerical')
        pl.plot(p1[:, 1], f_a[:, 1], '-.', color = 'black', label = 'Species-2 Analytic')

        pl.legend()
        pl.ylim([-1, 1])
        pl.title('Time =' + str(t0))
        pl.xlabel(r'$v$')
        pl.ylabel(r'$f$')
        pl.savefig('images/%04d'%((time_index+1)/4) + '.png')
        pl.clf()
