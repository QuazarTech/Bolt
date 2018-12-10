"""
File used to generate the dispersion plot
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

# Data used to create the dispersion plot:
k_numerical = np.arange(1, 30, 2)/10
whistler_numerical_values = np.array([0.10096006, 0.33288132, 0.5655006, 0.84127574, 1.1426955, 1.46512458, 1.80356678, 2.15296107, 2.50904493, 2.86628648, 3.22129582, 3.58019452, 3.91185725, 4.24244119, 4.56466215])
ion_cyclotron_numerical_values = np.array([0.08667781, 0.24211254, 0.37456496, 0.48163017, 0.56790879, 0.6371603, 0.69262697, 0.73703755, 0.77460004, 0.803542, 0.82690518, 0.84579796, 0.86108651, 0.87344762, 0.88341133])

# Analytic datapoints:
k = np.arange(1, 31) / 10
whistler = np.array([0.09749003842237096, 0.20270062656786197, 0.31564181352535126, 0.4362193812311734, 0.5642357758698509, 0.6993955075908691, 0.8413143822113407, 0.9895316400744226, 1.1435239413243987, 1.3027201476856392, 1.4665159765326323, 1.6342877991300446, 1.8054050770752403, 1.9792411437384998, 2.1551822180280933, 2.3326346755422174, 2.5110306960971576, 2.689832461811823, 2.868535104378185, 3.0466686024511223, 3.2237988180301382, 3.399527840572664, 3.573493784028673, 3.7453701581773835, 3.9148649135154927, 4.081719239514407, 4.245706179714867, 4.4066291138300375, 4.564320146512589, 4.718638434314387])
ion_cyclotron = np.array([0.0898800439758432, 0.17235295135132692, 0.24770193840644444, 0.31627594666517406, 0.378473560197061, 0.4347267201007696, 0.4854853296792793, 0.5312035805901814, 0.57232851937906, 0.6092910725675795, 0.6424994975912932, 0.672335046667114, 0.6991495234104763, 0.7232643677752562, 0.7449709077916147, 0.7645314498961183, 0.7821809289122078, 0.7981288930861946, 0.8125616518681449, 0.8256444603354378, 0.8375236525870926, 0.848328666898999, 0.8581739285999415, 0.8671605735852581, 0.8753780073148297, 0.8829053021650858, 0.8898124410947876, 0.8961614185402698, 0.902007210907606, 0.9073986294562573])

pl.plot(k, whistler, lw = 1, label = r'Whistler(Analytic)')
pl.plot(k, ion_cyclotron, lw = 1, color = 'C3', label = r'Alfv$\acute{e}$n(Analytic)')
pl.plot(k, k**0, '--', color = 'black')
pl.plot(k_numerical, whistler_numerical_values, 'x', color = 'C0', markersize = 10, label = r'Whistler(Numerical)')
pl.plot(k_numerical, ion_cyclotron_numerical_values, 'o', color = 'C3', markersize = 10, label = r'Alfv$\acute{e}$n(Numerical)')
pl.xlabel('$k(L^{-1})$')
pl.ylabel(r'$\omega((L / v_A)^{-1})$')
pl.xlim(np.min(k), np.max(k))
pl.ylim(0, np.max(whistler))
pl.legend(loc = 'upper left', bbox_to_anchor = (0., 1.05), fontsize = 20, framealpha = 0)
pl.savefig('plot.png', bbox_inches = 'tight')
