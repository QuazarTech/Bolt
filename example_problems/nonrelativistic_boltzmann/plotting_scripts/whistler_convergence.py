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

# Data for Whistler waves:
N = 32 + np.arange(9) * 16
error_B = np.array([1.97014590e-06, 1.63035102e-06, 1.17155987e-06, 8.53618088e-07,
                    6.40410253e-07, 4.93093452e-07, 3.89116050e-07, 3.13606925e-07, 2.57413619e-07])
error_E = np.array( [2.33079900e-06, 2.23653214e-06, 1.62631847e-06, 1.18243762e-06,
                     8.93780838e-07, 6.88299559e-07, 5.46538845e-07, 4.40388488e-07, 3.62006730e-07])
error_ve = np.array([4.39265247e-06, 3.60331737e-06, 2.61385515e-06, 1.91334337e-06,
                     1.43921355e-06, 1.10819172e-06, 8.73698386e-07, 7.04432859e-07, 5.78514159e-07])
error_vi = np.array([6.72319084e-07, 5.83753535e-07, 4.24682388e-07, 3.09231976e-07,
                     2.33128708e-07, 1.79804153e-07, 1.42232497e-07, 1.14674935e-07, 9.41934276e-08])

pl.loglog(N, error_ve, '-o', label = r'$v_e$')
pl.loglog(N, error_vi, '-o', label = r'$v_i$')
pl.loglog(N, error_E, '-o', label = r'$E$')
pl.loglog(N, error_B, '-o', label = r'$B$')
pl.loglog(N[-4:], 2e-2 / N[-4:]**2, '--', color = 'black', label = r'$\mathcal{O}(N^{-2})$')
pl.text(2**7, 1.3e-6, r'$\mathcal{O}(N^{-2})$', fontsize = 20)
pl.text(2**6 - 8, 3e-7, r'$p^+$', fontsize = 20)
pl.text(2**6 - 8, 3.2e-6, r'$e^-$', fontsize = 20)
pl.text(2**6 - 8, 9e-7, r'$B$', fontsize = 20)
pl.text(2**6 - 8, 2e-6, r'$E$', fontsize = 20)
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.xscale('log', basex=2)
pl.savefig('plot.png', bbox_inches = 'tight')
