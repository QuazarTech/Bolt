import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import domain as domain
import params as params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
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

def B3_analytic(q1, t):
    
    omega = -9.68977236633914e-18 - 0.009442719099991581*1j

    B3_analytic = params.amplitude * -0.16245984811645306 * \
                  np.exp(  1j * params.k_q1 * q1
                         + omega * t
                        ).real

    return(B3_analytic)

N = np.array([32, 48, 64, 96, 128])

error_B3 = np.zeros(N.size)

for i in range(N.size):

    dq1 = (domain.q1_end - domain.q1_start) / int(N[i])
    q1  = domain.q1_start + (np.arange(int(N[i]))) * dq1

    h5f = h5py.File('dump/N_%04d'%(int(N[i])) + '.h5')
    B3  = h5f['EM_fields'][:][0, :, 5]
    h5f.close()

    B3_ana = B3_analytic(q1, params.t_final)

    error_B3[i] = np.mean(abs(B3 - B3_ana))

print('Errors Obtained:')
print('L1 norm of error for B3:', error_B3)

print('\nConvergence Rates:')
print('Order of convergence for B3:', np.polyfit(np.log10(N), np.log10(error_B3), 1)[0])

# pl.loglog(N, error_n, '-o', label = 'Density')
# pl.loglog(N, error_v1, '-o', label = 'Velocity')
# pl.loglog(N, error_T, '-o', label = 'Temperature')
# pl.loglog(N, error_n[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
# pl.xlabel(r'$N$')
# pl.ylabel('Error')
# pl.legend()
# pl.savefig('convergenceplot.png')
