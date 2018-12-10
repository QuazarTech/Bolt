import numpy as np
import arrayfire as af
import matplotlib
# matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params
import initialize

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 80
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3

dq1 = (domain.q1_end - domain.q1_start) / N_q1
dq2 = (domain.q2_end - domain.q2_start) / N_q2

dp1 = (domain.p1_end[0] - domain.p1_start[0]) / N_p1
dp2 = (domain.p2_end[0] - domain.p2_start[0]) / N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * dp1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * dp2
p3 = np.array([0])

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

p2, p1, p3 = np.meshgrid(p2, p1, p3)
q2, q1     = np.meshgrid(q2, q1)

q1 = q1.reshape(1, N_q1, N_q2)
q2 = q2.reshape(1, N_q1, N_q2)
p1 = p1.reshape(N_p1 * N_p2 * N_p3, 1, 1)
p2 = p2.reshape(N_p1 * N_p2 * N_p3, 1, 1)
p3 = p3.reshape(N_p1 * N_p2 * N_p3, 1, 1)

# f0 = af.broadcast(initialize.initialize_f,
#                   af.to_array(q1 - p1 * 0), 
#                   af.to_array(q2 - p2 * 0), 
#                   af.to_array(p1), af.to_array(p2), af.to_array(p3), params
#                  )

# f0 = np.array(f0)
# f0 = f0.reshape(64, 64, 64, 64)

p1_t = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * dp1
q1_t = domain.q1_start    + (0.5 + np.arange(N_q1)) * dq1

q1_t, p1_t = np.meshgrid(q1_t, p1_t)

time_array = np.arange(0, 3 + 0.5, 
                       0.5
                      )

for time_index, t0 in enumerate(time_array):
    af.device_gc()

    q1_new = af.to_array(q1 - p1 * t0)
    q2_new = af.to_array(q2 - p2 * t0)

    # Periodic B.Cs
    for i in range(5):

        q1_new = af.select(q1_new < 0, q1_new + 1, q1_new)
        q2_new = af.select(q2_new < 0, q2_new + 1, q2_new)

        q1_new = af.select(q1_new > 1, q1_new - 1, q1_new)
        q2_new = af.select(q2_new > 1, q2_new - 1, q2_new)

    f = af.broadcast(initialize.initialize_f,
                     q1_new, q2_new, 
                     af.to_array(p1), af.to_array(p2), af.to_array(p3), params
                    )

    f = np.array(f)
    f = f.reshape(128, 32, 128, 32)

    pl.contourf(p1_t, q1_t, np.mean(abs(f), (1, 3)), np.linspace(0, 1.2, 150))
    # pl.colorbar()
    pl.xlabel(r'$v_x$')
    pl.ylabel(r'$x$')
    pl.title(r'Time = %.2f'%(t0))
    pl.savefig('images/%04d'%time_index + '.png', bbox_inches = 'tight')
    # pl.savefig('plot.png')
    pl.clf()

q1, p1 = np.meshgrid(q1, p1)

# Declaration of the time array:
time_array = np.arange(0, params.t_final + 0.01, 
                       0.01
                      )

f_max = -1e100
f_min =  1e100

for time_index, t0 in enumerate(time_array):
    print('t =', t0)

    h5f = h5py.File('dump_f/t=%.3f'%(t0) + '.h5', 'r')
    f   = np.mean(np.swapaxes(h5f['distribution_function'][:], 0, 1).reshape(N_q1, N_q2, N_p1, N_p2), (1, 2))
    h5f.close()

    if(f.max() > f_max):
        f_max = abs(f).max()

    if(f.min() < f_min):
        f_min = abs(f).min()

for time_index, t0 in enumerate(time_array):
    
    h5f = h5py.File('dump_f/t=%.3f'%(t0) + '.h5', 'r')
    f   = np.mean(np.swapaxes(h5f['distribution_function'][:], 0, 1).reshape(N_q1, N_q2, N_p1, N_p2), (1, 2))
    h5f.close()

    pl.contourf(q1, p1, abs(f), 100) #np.linspace(f_min, f_max, 100))
    # pl.colorbar()
    pl.xlabel(r'$v_x$')
    pl.ylabel(r'$x$')
    pl.title('Time = %.2f'%(t0))
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()
