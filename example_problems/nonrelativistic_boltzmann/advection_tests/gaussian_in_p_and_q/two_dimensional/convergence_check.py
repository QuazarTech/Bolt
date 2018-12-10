import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

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

N = np.array([32, 48, 64, 96, 112]) #, 128, 144, 160])

def check_error(params):

    error = np.zeros(N.size)

    for i in range(N.size):
        af.device_gc()
        print(N[i])
        N_q1 = int(N[i])
        N_q2 = int(N[i])
        N_p1 = int(N[i])
        N_p2 = int(N[i])
        N_p3 = 1
        N_g  = domain.N_ghost

        dq1 = (domain.q1_end - domain.q1_start) / N_q1
        dq2 = (domain.q2_end - domain.q2_start) / N_q2

        dp1 = (domain.p1_end[0] - domain.p1_start[0]) / N_p1
        dp2 = (domain.p2_end[0] - domain.p2_start[0]) / N_p2
        dp3 = (domain.p3_end[0] - domain.p3_start[0]) / N_p3

        q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
        q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

        p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * dp1
        p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * dp2
        p3 = domain.p3_start[0] + (0.5 + np.arange(N_p3)) * dp3

        p2, p1, p3 = np.meshgrid(p2, p1, p3)
        q2, q1     = np.meshgrid(q2, q1)

        q1 = q1.reshape(1, N_q1, N_q2)
        q2 = q2.reshape(1, N_q1, N_q2)
        p1 = p1.reshape(N_p1 * N_p2 * N_p3, 1, 1)
        p2 = p2.reshape(N_p1 * N_p2 * N_p3, 1, 1)
        p3 = p3.reshape(N_p1 * N_p2 * N_p3, 1, 1)

        h5f = h5py.File('dump/%04d'%(int(N[i])) + '.h5', 'r')
        f   = np.swapaxes(np.swapaxes(h5f['distribution_function'][:], 0, 2), 1, 2)
        h5f.close()

        q1_new = af.to_array(q1 - p1 * params.t_final)
        q2_new = af.to_array(q2 - p2 * params.t_final)

        # Periodic B.Cs
        for j in range(5):

            q1_new = af.select(q1_new < 0, q1_new + 1, q1_new)
            q2_new = af.select(q2_new < 0, q2_new + 1, q2_new)

            q1_new = af.select(q1_new > 1, q1_new - 1, q1_new)
            q2_new = af.select(q2_new > 1, q2_new - 1, q2_new)

        f_reference = af.broadcast(initialize.initialize_f,
                                   q1_new, q2_new, 
                                   af.to_array(p1), af.to_array(p2), af.to_array(p3), params
                                  )

        error[i] = np.mean(abs(f - np.array(f_reference)))

    return(error)

err = check_error(params)
con = np.polyfit(np.log10(N), np.log10(err), 1)

print('Error:', err)
print('Convergence Rate:', con[0])

pl.loglog(N, err, '-o', label = 'Numerical')
pl.loglog(N, err[0] * 32/N, '--', color = 'black', label = r'$O(N^{-1})$')
pl.loglog(N, err[0] * 32**2/N**2, '-.', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
