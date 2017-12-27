import numpy as np
import h5py
from scipy.integrate import odeint
import pylab as pl

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 80
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

charge = -10
mass   = 1

E1 = 2
E2 = 3
B3 = 1.8

dt      = 0.001
t_final = 1.0

t = np.arange(0, t_final + dt, dt)
p = np.zeros(400)

def dpdt(p, t):
    p1 = p[:200]
    p2 = p[200:]

    dp1_dt = (charge/mass) * (E1 + p2 * B3)
    dp2_dt = (charge/mass) * (E2 - p1 * B3)
    dp_dt  = np.append(dp1_dt, dp2_dt)
    
    return(dp_dt)

lims = 0.1 * np.arange(1, 11)

for i in range(lims.size): 
    # getting a circle
    p1 = -lims[i] + (0.5 + np.arange(10)) * lims[i]/5
    p2 = np.sqrt(lims[i]**2 - p1**2)

    for j in range(20):
        if(j<10):
            p[20*i + j]       = p1[j]
            p[200 + 20*i + j] = p2[j]
        else:
            p[20*i + j]       = p1[j-10]
            p[200 + 20*i + j] = -p2[j-10]

sol  = odeint(dpdt, p, t)
diff = np.sum(abs(sol[1:] - sol[0]), 1)

h5f = h5py.File('particle_traj.h5', 'w')
h5f.create_dataset('particle_paths', data = sol)
h5f.close()

# for t0 in range(t.size):

#     for i in range(100):
#         pl.plot(sol[t0, i], sol[t0, i + 200], 'or')
    
#     pl.xlim(-1.5, 1.5)
#     pl.ylim(-1.5, 1.5)
#     pl.title('Time = %.2f'%(t[t0]-dt))
#     pl.grid()
#     pl.gca().set_aspect('equal')
#     pl.xlabel(r'$p_1$')
#     pl.ylabel(r'$p_2$')
#     pl.savefig('images/%04d'%t0 + '.png')
#     pl.clf()
