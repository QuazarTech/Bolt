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

dt      = 0.005
t_final = 1.0
time    = np.arange(0, t_final + dt, dt)

N_q1 = domain.N_q1
N_q2 = domain.N_q2

N_p1 = domain.N_p1
N_p2 = domain.N_p2

r     = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start) / N_q1
theta = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start) / N_q2

theta, r = np.meshgrid(theta, r)

r     = r.reshape(N_q1, N_q2, 1, 1)
theta = theta.reshape(N_q1, N_q2, 1, 1)

rdot     = domain.p1_start + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - domain.p1_start[0]) / N_p1
thetadot = domain.p2_start + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - domain.p2_start[0]) / N_p2

thetadot, rdot = np.meshgrid(thetadot, rdot)

rdot     = rdot.reshape(1, 1, N_p1, N_p2)
thetadot = thetadot.reshape(1, 1, N_p1, N_p2)

xdot = rdot * np.cos(theta) - r * np.sin(theta) * thetadot
ydot = rdot * np.sin(theta) + r * np.cos(theta) * thetadot

xdot = np.sum(np.sum(xdot, 0), 1).reshape(domain.N_p1, domain.N_p2)
ydot = np.sum(np.sum(ydot, 0), 1).reshape(domain.N_p1, domain.N_p2)

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index) + '.h5', 'r')
    f   = np.sum(np.sum(np.swapaxes(h5f['distribution_function'][:], 0, 1), 0), 1).reshape(domain.N_p1, domain.N_p2)
    h5f.close()

    pl.contourf(xdot, ydot, n, 200)
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$\dot{x}$')
    pl.ylabel(r'$\dot{y}$')
    pl.title('Time = %.3f'%t0)
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()
