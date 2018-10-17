"""
File used to create the two stream electric field
evolution plot used in the bolt code paper
"""

import numpy as np
import pylab as pl
import h5py

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

h5f = h5py.File('collisional.h5', 'r')
E1  = h5f['electrical_field_nls'][:]
t1  = h5f['time'][:]
h5f.close()

h5f = h5py.File('collisionless.h5', 'r')
E2  = h5f['electrical_field_nls'][:]
t2  = h5f['time'][:]
h5f.close()

pl.semilogy(t2, E2, label = r'$\tau = \infty$')
pl.semilogy(t1, E1, color = 'C3', label = r'$\tau = 0.001(\omega_p^{-1})$')
pl.xlabel(r'Time$(\omega_p^{-1})$')
pl.ylabel(r'SUM$(|E|^2)$')
pl.legend(framealpha = 0, fontsize = 28, loc = 'center right', bbox_to_anchor = [1, 0.75])
pl.savefig('plot.png', bbox_inches = 'tight')
