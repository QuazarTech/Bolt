import numpy as np
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 10, 14
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
t_final = 0.2
time    = np.arange(dt, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

h5f  = h5py.File('dump/0000.h5', 'r')
q1   = h5f['q1'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
q2   = h5f['q2'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
n    = h5f['n'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
p1   = h5f['p1'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
T    = h5f['T'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
h5f.close()

fig = pl.figure()

ax1 = fig.add_subplot(3,1,1)
ax1.plot(q1[N_g:-N_g, N_g], n[N_g:-N_g, N_g])
ax1.set_ylabel(r'$\rho$')

ax2 = fig.add_subplot(3,1,2)
ax2.plot(q1[N_g:-N_g, N_g], p1[N_g:-N_g, N_g])
ax2.set_ylabel(r'$v_x$')
ax2.set_ylim([0, 1])

ax3 = fig.add_subplot(3,1,3)
ax3.plot(q1[N_g:-N_g, N_g], n[N_g:-N_g, N_g] * T[N_g:-N_g, N_g])
ax3.set_ylabel(r'$p$')
ax3.set_xlabel('$x$')

fig.suptitle('Time = 0')
pl.savefig('images/0000.png')
pl.close(fig)
pl.clf()

for time_index, t0 in enumerate(time):
    
    h5f  = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    n    = h5f['n'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
    p1   = h5f['p1'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
    T    = h5f['T'][:].reshape(N_q1 + 2 * N_g, N_q2 + 2 * N_g)
    h5f.close()

    fig = pl.figure()

    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(q1[N_g:-N_g, N_g], n[N_g:-N_g, N_g])
    ax1.set_ylabel(r'$\rho$')

    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(q1[N_g:-N_g, N_g], p1[N_g:-N_g, N_g])
    ax2.set_ylabel(r'$v_x$')
    ax2.set_ylim([0, 1])

    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(q1[N_g:-N_g, N_g], n[N_g:-N_g, N_g] * T[N_g:-N_g, N_g])
    ax3.set_ylabel(r'$p$')
    ax3.set_xlabel('$x$')

    fig.suptitle('Time = ' + str(t0))
    pl.savefig('images/' + '%04d'%time_index + '.png')
    pl.close(fig)
    pl.clf()
