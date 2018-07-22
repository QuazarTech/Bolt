import arrayfire as af
import numpy as np
import h5py
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

N = 2**np.arange(5, 10)

def check_error(params):

    error = np.zeros(N.size)

    for i in range(N.size):

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

        q2, q1, p1, p2, p3 = np.meshgrid(q2, q1, p1, p2, p3)

        q1 = q1.reshape(domain.N_q1, domain.N_q2, domain.N_p1 * domain.N_p2 * domain.N_p3)
        q2 = q2.reshape(domain.N_q1, domain.N_q2, domain.N_p1 * domain.N_p2 * domain.N_p3)
        p1 = p1.reshape(domain.N_q1, domain.N_q2, domain.N_p1 * domain.N_p2 * domain.N_p3)
        p2 = p2.reshape(domain.N_q1, domain.N_q2, domain.N_p1 * domain.N_p2 * domain.N_p3)
        p3 = p3.reshape(domain.N_q1, domain.N_q2, domain.N_p1 * domain.N_p2 * domain.N_p3)
        
        q1 = af.to_array(q1)
        q2 = af.to_array(q2)
        p1 = af.to_array(p1)
        p2 = af.to_array(p2)
        p3 = af.to_array(p3)

        h5f = h5py.File('dump/%04d'%(int(N[i])) + '.h5', 'r')
        f   = af.to_array(np.flip(np.swapaxes(h5f['distribution_function'][:], 0, 1), 2))
        h5f.close()

        f_reference = af.broadcast(initialize.initialize_f,
                                   q1 - p1 * params.t_final, 
                                   q2 - p2 * params.t_final, 
                                   p1, p2, p3, params
                                  )

        error[i] = af.mean(af.abs(f - f_reference))

    return(error)

err = check_error(params)
con = np.polyfit(np.log10(N), np.log10(err), 1)

print('Convergence Rate:', con[0])

pl.loglog(N, err, '-o', label = 'Numerical')
pl.loglog(N, err[0] * 32/N, '--', color = 'black', label = r'$O(N^{-1})$')
pl.loglog(N, err[0] * 32**2/N**2, '-.', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
