import numpy as np
from scipy.integrate import odeint
import pylab as pl

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

N_p1 = 96
N_p2 = 96

p1_start = -10
p1_end   = 10
dp1      = (p1_end - p1_start) / N_p1 

p2_start = -10
p2_end   = 10
dp2      = (p2_end - p2_start) / N_p2

p1 = p1_start + (0.5 + np.arange(N_p1)) * dp1
p2 = p2_start + (0.5 + np.arange(N_p2)) * dp2

p2, p1 = np.meshgrid(p2, p1)

def dfdt(f, t, A_p1, A_p2):
    f_reshaped = f.reshape(N_p1, N_p2)
    dfdp1      = np.gradient(f_reshaped, axis = 0) / dp1
    dfdp2      = np.gradient(f_reshaped, axis = 1) / dp2
    
    dfdt = - A_p1 * dfdp1.ravel() - A_p2 * dfdp2.ravel()
    
    return dfdt

f_initial =   (1 / (2 * np.pi)) * np.exp (-0.5 * p1**2) * np.exp (-0.5 * p2**2)

E1 = 2
E2 = 3
B3 = 1.8

charge = -10

A_p1 = charge * (E1 + p2 * B3).ravel()
A_p2 = charge * (E2 - p1 * B3).ravel()

dt      = 0.001 * 32 / N_p1
t_final = 0.5

t = np.arange(dt, t_final + dt, dt)

sol = odeint(dfdt, f_initial.ravel(), t, args = (A_p1, A_p2))

for i in range(sol.shape[0]):
    
    pl.contourf(p1, p2, sol[i].reshape(N_p1, N_p2), 100, cmap = 'bwr')
    pl.gca().set_aspect('equal')
    pl.title('Time =%.3f'%(i * dt))
    pl.savefig('images/%04d'%i + '.png')
    pl.clf()
