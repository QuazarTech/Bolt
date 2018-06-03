import numpy as np
import pylab as pl
import h5py
import domain
from scipy.integrate import odeint

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
pl.rcParams['figure.dpi']      = 300
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

def dpdt(p, t, E1, E2, B3, charge, mass):
    p1 = p[0]
    p2 = p[1]

    dp1_dt = (charge/mass) * (E1 + p2 * B3)
    dp2_dt = (charge/mass) * (E2 - p1 * B3)
    dp_dt  = np.append(dp1_dt, dp2_dt)
    return(dp_dt)

dt      = 0.001
t_final = 0.5
time    = np.arange(dt, t_final + dt, dt)

h5f = h5py.File('dump/0000.h5', 'r')
p1  = h5f['p1'][:].reshape(128, 128)
p2  = h5f['p2'][:].reshape(128, 128)
fi  = h5f['distribution_function'][:].reshape(128, 128)
h5f.close()

maxf = np.max(fi) + 0.02
minf = np.min(fi) - 0.02

sol = odeint(dpdt, np.array([0, 0]), time,
             args = (2, 3, 1.8, -10, 1)
            ) 

for time_index, t0 in enumerate(time):
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
    p1  = h5f['p1'][:].reshape(128, 128)
    p2  = h5f['p2'][:].reshape(128, 128)
    f   = h5f['distribution_function'][:].reshape(128, 128)
    h5f.close()

pl.xlim(-7, 7)
pl.ylim(-7, 7)
pl.xlabel(r'$v_x(E_0 / B_0)$')
pl.ylabel(r'$v_y(E_0 / B_0)$')
pl.axes().set_aspect('equal')
pl.plot(sol[:, 0], sol[:, 1], color = 'white', linewidth = 5, alpha = 0.3)
pl.contourf(p1, p2, fi, np.linspace(minf, maxf, 150))
pl.savefig('plot.png')
pl.clf()
