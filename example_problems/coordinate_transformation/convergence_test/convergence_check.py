import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

import domain
import params
import example_problems.nonrelativistic_boltzmann.advection_tests.gaussian_in_p_and_q.two_dimensional.initialize \
       as initialize

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
        N_r        = int(N[i])
        N_theta    = int(N[i])
        N_rdot     = int(N[i])
        N_thetadot = int(N[i])
        N_p3       = 1
        N_g        = domain.N_ghost

        dr        = (domain.q1_end - domain.q1_start) / N_r
        dtheta    = (domain.q2_end - domain.q2_start) / N_theta

        drdot     = (domain.p1_end[0] - domain.p1_start[0]) / N_rdot
        dthetadot = (domain.p2_end[0] - domain.p2_start[0]) / N_thetadot
        dp3       = (domain.p3_end[0] - domain.p3_start[0]) / N_p3

        r     = domain.q1_start + (0.5 + np.arange(N_r)) * dr
        theta = domain.q2_start + (0.5 + np.arange(N_theta)) * dtheta

        rdot     = domain.p1_start[0] + (0.5 + np.arange(N_rdot)) * drdot
        thetadot = domain.p2_start[0] + (0.5 + np.arange(N_thetadot)) * dthetadot
        p3       = domain.p3_start[0] + (0.5 + np.arange(N_p3)) * dp3

        thetadot, rdot, p3 = np.meshgrid(thetadot, rdot, p3)
        theta, r           = np.meshgrid(theta, r)

        r        = r.reshape(1, N_r, N_theta)
        theta    = theta.reshape(1, N_r, N_theta)
        rdot     = rdot.reshape(N_rdot * N_thetadot * N_p3, 1, 1)
        thetadot = thetadot.reshape(N_rdot * N_thetadot * N_p3, 1, 1)
        p3       = p3.reshape(N_rdot * N_thetadot * N_p3, 1, 1)

        q1 = r * np.cos(theta)
        q2 = r * np.sin(theta)

        p1 = rdot * np.cos(theta) - r * np.sin(theta) * thetadot
        p2 = rdot * np.sin(theta) + r * np.cos(theta) * thetadot

        f_reference = af.broadcast(initialize.initialize_f,
                                   af.to_array(q1 - p1 * params.t_final), 
                                   af.to_array(q2 - p2 * params.t_final), 
                                   af.to_array(p1), af.to_array(p2), af.to_array(p3), params
                                  )

        h5f = h5py.File('dump/%04d'%(int(N[i])) + '.h5', 'r')
        f   = np.swapaxes(np.swapaxes(h5f['distribution_function'][:], 0, 2), 1, 2)
        h5f.close()

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
