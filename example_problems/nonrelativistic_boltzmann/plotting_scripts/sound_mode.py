"""
File used to generate the convergence plot for
sound wave convergence in the bolt code paper
"""

import numpy as np
import pylab as pl

# Plot parameters:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['figure.dpi']      = 300
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

N = np.array([32, 48, 64, 96, 128, 144, 160])

# Error analytic:
error_n = np.array([1.79769166e-06, 8.46297727e-07, 4.86982505e-07, 2.09151707e-07, 
                    1.39955381e-07, 1.21192482e-07, 1.10273088e-07])
error_v = np.array([3.04872891e-06, 1.43156800e-06, 8.21989502e-07, 3.80489042e-07, 
                    2.74536537e-07, 2.44205841e-07, 2.27109353e-07])
error_T = np.array([3.38503758e-06, 1.63201639e-06, 9.41241128e-07, 4.26936480e-07, 
                    3.04857305e-07, 2.69862536e-07, 2.50240630e-07])

pl.loglog(N, error_n, '-o', label = 'Density')
pl.loglog(N, error_v, '-o', label = 'Velocity')
pl.loglog(N, error_T, '-o', label = 'Temperature')
pl.loglog(N, (error_n[0] + 1e-7) * 32**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_x^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_x$')
pl.ylabel('Error')
pl.text(140, 1.5e-7, r'$\mathcal{O}(\tau)$', fontsize = 20)
pl.fill_between(N[-3:], error_n[-3:], error_v[-3:], color = 'black', alpha = 0.2)
pl.fill_between(N[-3:], error_v[-3:], error_T[-3:], color = 'black', alpha = 0.2)
pl.legend(loc = 'upper right', fontsize = 20, ncol = 2, framealpha = 0, bbox_to_anchor = (1.0, 1.05))
pl.savefig('plot.png', bbox_inches = 'tight')

# Error with linear solver:
error_n = np.array([1.79544106e-06, 8.46596256e-07, 4.87296318e-07, 2.03563625e-07,
                    1.15409808e-07, 8.85741505e-08, 7.20919939e-08])
error_v = np.array([3.04463900e-06, 1.43169454e-06, 8.22316443e-07, 3.47568222e-07,
                    1.97828164e-07, 1.51363362e-07, 1.23495737e-07])
error_T = np.array([3.37397375e-06, 1.63211191e-06, 9.41032336e-07, 3.96564288e-07,
                    2.25866784e-07, 1.73336634e-07, 1.41329111e-07])

pl.loglog(N, error_n, '-o', label = 'Density')
pl.loglog(N, error_v, '-o', label = 'Velocity')
pl.loglog(N, error_T, '-o', label = 'Temperature')
pl.loglog(N, (error_n[0] + 1e-7) * 32**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_x^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_x$')
pl.ylabel('Error')
# pl.legend(loc = 'upper right', fontsize = 20, ncol = 2, framealpha = 0, bbox_to_anchor = (1.0, 1.05))
pl.savefig('plot2.png', bbox_inches = 'tight')
