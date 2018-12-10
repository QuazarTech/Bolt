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

N = 2**np.arange(6, 11)
# With minmod:
error_periodic = np.array([3.84e-4, 1.74e-4, 5.49e-5, 1.51e-5, 4.28e-6])
error_mirror   = np.array([3.84e-4, 1.74e-4, 5.49e-5, 1.51e-5, 4.28e-6])

pl.loglog(N, error_periodic, '-o', label = 'Periodic')
pl.loglog(N, error_mirror, '--o', color = 'C3', label = 'Mirror')
pl.loglog(N, (error_periodic[0] + 1e-3) * 64**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_x^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_x$')
pl.ylabel('Error')
pl.legend(fontsize = 28, framealpha = 0)
# pl.legend(loc = 'upper right', bbox_to_anchor=(1.05, 1.15), ncol=1, fancybox=True, shadow=True)
pl.savefig('plot.png', bbox_inches = 'tight')
