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

N = np.array([32, 48, 64, 96, 128])

error_2V = np.array([5.25e-4, 2.71e-4, 1.63e-4, 7.44e-5, 4.36e-5])
error_3V = np.array([4.7e-5, 2.2e-5, 1.16e-5, 2.63e-6, 7.67e-7])

pl.loglog(N, error_2V, '-o', label = '2V')
pl.loglog(N, error_3V, '-o', color = 'C3', label = '3V')
pl.loglog(N, error_2V[0] * 32**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_v^{-2})$')
pl.loglog(N, error_3V[0] * 32**2/N**2, '--', color = 'black')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_v$')
pl.ylabel('Error')
lgd = pl.legend(fontsize = 28, framealpha = 0, bbox_to_anchor = (0.4, 0.65))
pl.savefig('plot.png',bbox_extra_artists=(lgd,), bbox_inches = 'tight')
