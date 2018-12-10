import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
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

def dp_dt(p, t, E1, E2, E3, B1, B2, B3, charge, mass):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    dp1_dt = (charge/mass) * (E1 + p2 * B3 - p3 * B2)
    dp2_dt = (charge/mass) * (E2 + p3 * B1 - p1 * B3)
    dp3_dt = (charge/mass) * (E3 + p1 * B2 - p2 * B1)

    dp_dt  = np.array([dp1_dt, dp2_dt, dp3_dt])
    return(dp_dt)

dt      = 0.001
t_final = 0.2
time    = np.arange(dt, t_final + dt, dt)

sol = odeint(dp_dt, np.array([0, 0, 0]), time,
             args = (0.01, 0.002, 0.0003, 0.4, 10.5, 20.6, -10, 1),
             atol = 1e-12, rtol = 1e-12
            ) 


ax = pl.axes(projection='3d')

# tmp_planes = ax.zaxis._PLANES 
# ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
#                      tmp_planes[0], tmp_planes[1], 
#                      tmp_planes[4], tmp_planes[5])
# view_1 = (25, -135)
# view_2 = (25, -45)
# init_view = view_2
# ax.view_init(*init_view)

ax.plot3D(sol[:, 0], sol[:, 1], sol[:, 2])
ax.plot([sol[0, 0]], [sol[0, 1]], [sol[0, 2]], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=5)
ax.plot([sol[-1, 0]], [sol[-1, 1]], [sol[-1, 2]], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=5)
ax.set_xlabel(r'$v_x$', linespacing=10, labelpad=30)
ax.set_ylabel(r'$v_y$', linespacing=10, labelpad=30)
ax.set_zlabel(r'$v_z$', linespacing=15, labelpad=30)

# ax.yaxis.set_major_formatter(pl.NullFormatter())
# ax.xaxis.set_major_formatter(pl.NullFormatter())

ax.xaxis.set_major_locator(pl.MaxNLocator(4))
ax.yaxis.set_major_locator(pl.MaxNLocator(4))
ax.zaxis.set_major_locator(pl.MaxNLocator(4))

pl.savefig('plot.png')
pl.clf()
