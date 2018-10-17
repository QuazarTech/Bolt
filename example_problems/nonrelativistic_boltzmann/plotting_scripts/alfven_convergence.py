"""
File used to generate convergence plot
for alfven wave convergence
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

N = 32 + np.arange(9) * 16

error_B = np.array([1.88239359e-06, 1.34705322e-06, 9.88411219e-07, 7.16857390e-07,
                    5.33225860e-07, 4.09392308e-07, 3.22725226e-07, 2.60369317e-07, 2.13938745e-07])
error_E = np.array([7.20737970e-07, 4.85519129e-07, 3.65979532e-07, 2.64070151e-07,
                    1.93870265e-07, 1.47482986e-07, 1.14747226e-07, 9.29497498e-08, 7.59843193e-08])
error_ve = np.array([6.84474776e-07, 4.39461983e-07, 3.23219057e-07, 2.25775400e-07, 
                     1.64093065e-07, 1.26171326e-07, 9.91391890e-08, 7.91353527e-08, 6.54439517e-08])
error_vi = np.array([4.34199278e-06, 3.12494787e-06, 2.28197680e-06, 1.65532127e-06,
                     1.22502294e-06, 9.39663685e-07, 7.40306682e-07, 5.96650924e-07, 4.90344023e-07])

pl.loglog(N, error_ve, '-o', label = r'$v_e$')
pl.loglog(N, error_vi, '-o', label = r'$v_i$')
pl.loglog(N, error_E, '-o', label = r'$E$')
pl.loglog(N, error_B, '-o', label = r'$B$')
pl.loglog(N[-4:], 2e-2 / N[-4:]**2, '--', color = 'black', label = r'$\mathcal{O}(N^{-2})$')
pl.text(2**7, 1.3e-6, r'$\mathcal{O}(N^{-2})$', fontsize = 20)
pl.text(2**6 - 8, 2.5e-7, r'$e^-$', fontsize = 20)
pl.text(2**6 - 8, 3.2e-6, r'$p^+$', fontsize = 20)
pl.text(2**6 - 8, 4.5e-7, r'$E$', fontsize = 20)
pl.text(2**6 - 8, 1.3e-6, r'$B$', fontsize = 20)
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.xscale('log', basex=2)
pl.savefig('plot.png', bbox_inches = 'tight')
