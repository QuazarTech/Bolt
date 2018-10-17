"""
File used to generate the convergence plot for
langmuir wave convergence in the bolt code paper
"""

import numpy as np
import pylab as pl

# Plot parameters:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['figure.dpi']      = 300
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

N     = np.array([128, 192, 256, 384, 512])
error = np.array([7.81702889e-10, 3.54381043e-10, 2.00960529e-10, 8.99856395e-11, 5.07863474e-11])

pl.loglog(N, error, '-o', label = 'Numerical')
pl.loglog(N, error[0] * 128**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend(fontsize = 28, framealpha = 0)
# pl.legend(loc = 'upper right', bbox_to_anchor=(1.05, 1.15), ncol=1, fancybox=True, shadow=True)
pl.savefig('plot.png', bbox_inches = 'tight')
