import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
# mpl.use('agg')
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

N_particles = 100

def dX_dt(X, t):
    r        = X[:N_particles]
    theta    = X[N_particles:2 * N_particles]
    rdot     = X[2 * N_particles:3 * N_particles]
    thetadot = X[3 * N_particles:4 * N_particles]

    dr_dt        = rdot
    dtheta_dt    = thetadot
    drdot_dt     = r * thetadot**2
    dthetadot_dt = -2 * rdot * thetadot / r

    return(np.concatenate([dr_dt, dtheta_dt, drdot_dt, dthetadot_dt]))

dt      = 0.0005
t_final = 1.0
time    = np.arange(0, t_final + dt, dt)

# Analytic solution:
r0 = 1.5
dr = 0.5

r_init        = r0 + dr * (2 * np.random.rand(N_particles) - 1)
theta_init    = np.zeros(N_particles)
rdot_init     = 1.015625 * np.ones(N_particles)
thetadot_init = 1 / r_init

# Storing the above data into a single vector:
X0 = np.concatenate([r_init, theta_init, rdot_init, thetadot_init])
sol = odeint(dX_dt, X0, time)

r        = sol[:, :N_particles]
theta    = sol[:, 1 * N_particles:2 * N_particles]
rdot     = sol[:, 2 * N_particles:3 * N_particles]
thetadot = sol[:, 3 * N_particles:4 * N_particles]

# Getting results in cartesian coordinates:
x_analytic = r * np.cos(theta)
y_analytic = r * np.sin(theta)

N_q1 = domain.N_q1
N_q2 = domain.N_q2

h5f   = h5py.File('dump/0000.h5', 'r')
r     = h5f['q1'][:].reshape(N_q1, N_q2)
theta = h5f['q2'][:].reshape(N_q1, N_q2)
n0    = h5f['n'][:].reshape(N_q1, N_q2)
h5f.close()

x = r * np.cos(theta)
y = r * np.sin(theta)

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index) + '.h5', 'r')
    n   = h5f['n'][:].reshape(N_q1, N_q2)
    h5f.close()

    pl.plot(x_analytic[time_index], y_analytic[time_index], 'o', color = 'white', alpha = 0.05)

    import arrayfire as af
    n = np.array(af.select(af.to_array(n)<0, 0, af.to_array(n)))

    pl.contourf(x, y, n, 200)
    pl.gca().set_aspect('equal')
    pl.xlim(0, 4)
    pl.ylim(-4, 4)
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.title('Time = %.3f'%t0)
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()

# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(x_analytic[time_index+1], y_analytic[time_index+1], 'or')
# ax2.set_xlim(0, 4)
# ax2.set_ylim(-4, 4)
# ax2.set_xlabel(r'$x$')
# ax2.set_ylabel(r'$y$')
