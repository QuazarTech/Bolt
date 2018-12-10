"""
File used to generate the convergence plot for EM solver
in the bolt code paper
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

N = 2**np.arange(5, 10)

error_E1 = np.array([ 0.22239706,  0.06520287,  0.01776956,  0.00464663,  0.00118863])
error_E2 = np.array([ 0.11138749,  0.03260836,  0.00888502,  0.00232332,  0.00059431])
error_B3 = np.array([ 0.2483709,   0.07288678,  0.0198664,   0.00519507,  0.00132892])

error_B1 = np.array([ 0.20643534,  0.05890246,  0.01593439,  0.00415878,  0.00106331])
error_B2 = np.array([ 0.10348738,  0.02947937,  0.00796937,  0.00207954,  0.00053167])
error_E3 = np.array([ 0.23331283,  0.06596588,  0.0178202,   0.0046499,   0.00118883])

pl.loglog(N, error_E1, '-o', label = r'$E_x$')
pl.loglog(N, error_E2, '-o', color = 'C3', label = r'$E_y$')
pl.loglog(N, error_B3, '-o', color = 'C2', label = r'$B_z$')
pl.loglog(N, error_E1[0] * 32**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_x^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_x$')
pl.ylabel('Error')
lgd = pl.legend(loc = 'lower left', fontsize = 28, framealpha = 0, bbox_to_anchor = [-0.04, -0.12])
pl.savefig('plot.png',bbox_extra_artists=(lgd,), bbox_inches = 'tight')
pl.clf()

pl.loglog(N, error_B1, '-o', label = r'$B_x$')
pl.loglog(N, error_B2, '-o', color = 'C3', label = r'$B_y$')
pl.loglog(N, error_E3, '-o', color = 'C2', label = r'$E_z$')
pl.loglog(N, error_B1[0] * 32**2/N**2, '--', color = 'black', label = r'$\mathcal{O}(N_x^{-2})$')
pl.xscale('log', basex=2)
pl.xlabel(r'$N_x$')
pl.ylabel('Error')
lgd = pl.legend(loc = 'lower left', fontsize = 28, framealpha = 0, bbox_to_anchor = [-0.04, -0.12])
pl.savefig('plot2.png',bbox_extra_artists=(lgd,), bbox_inches = 'tight')
pl.clf()
