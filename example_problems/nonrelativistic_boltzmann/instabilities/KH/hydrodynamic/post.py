import numpy as np
import pylab as pl
import h5py

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

dt      = 0.0001
t_final = 1.0
time    = np.arange(dt, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

q1 = (0.5 + np.arange(N_q1)) * 1/N_q1
q2 = (0.5 + np.arange(N_q2)) * 2/N_q2

q2, q1 = np.meshgrid(q2, q1)

h5f     = h5py.File('dump/0000.h5', 'r')
moments = np.swapaxes(h5f['moments'][:], 0, 1)
h5f.close()

n = moments[:, :, 0]

pl.contourf(q1, q2, n, 100)
pl.title('Time = 0')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.colorbar()
pl.savefig('images/0000.png')
pl.clf()

for time_index, t0 in enumerate(time):
    
    if((time_index+1)%10 == 0):

        h5f  = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()
        
        n = moments[:, :, 0]

        pl.contourf(q1, q2, n, 100)
        pl.title('Time = ' + "%.2f"%(t0))
        pl.xlabel(r'$x$')
        pl.ylabel(r'$y$')
        pl.colorbar()
        pl.savefig('images/%04d'%(time_index+1) + '.png')
        pl.clf()
