import numpy as np
from scipy.integrate import odeint
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

def vel(x, t):
    return(np.array([2, 1]))

dt      = 0.0005
t_final = 0.9
time    = np.arange(dt, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2

h5f = h5py.File('dump/0000.h5', 'r')
q1  = h5f['q1'][:].reshape(N_q1, N_q2)
q2  = h5f['q2'][:].reshape(N_q1, N_q2)
n0  = h5f['n'][:].reshape(N_q1, N_q2)
h5f.close()

pl.contourf(q1, q2, n0, 100)
pl.title('Time = 0')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.axes().set_aspect('equal')
pl.savefig('images/0000.png')
pl.clf()

traj = odeint(vel, np.array([0.5, 0.5]), time)

x = traj[:, 0]
y = traj[:, 1]

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    n   = h5f['n'][:].reshape(N_q1, N_q2)
    h5f.close()

x = np.where(x>1, x-1, x)
x = np.where(x>1, x-1, x)
y = np.where(y>1, y-1, y)

x = np.where(x<0, x+1, x)
y = np.where(y<0, y+1, y)

pos = np.where(np.abs(np.diff(y)) >= 0.5)[0]

x[pos] = np.nan
y[pos] = np.nan

pos = np.where(np.abs(np.diff(x)) >= 0.5)[0]

x[pos] = np.nan
y[pos] = np.nan

nf = n0 + n
pl.contourf(q1, q2, nf, 100)
# pl.contourf(q1, q2, n, 50, alpha = 0.5)
pl.plot(x, y, linewidth = 5, color = 'white', alpha = 0.3)
# pl.title('Time =' + str(t0))
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.xlim([0, 1])
pl.ylim([0, 1])
pl.axes().set_aspect('equal')
        # pl.savefig('images/%04d'%((time_index+1)/4) + '.png')
        # pl.clf()

# pl.savefig('trial.svg')
pl.savefig('trial.png')

