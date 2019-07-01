import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import domain
import params

from post import p1, p2, p3, return_f

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
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


# Getting the distribution function at t = 0:
# Since we want to visualize variation in (p1, p2), we will be averaging all other quantities:
# (q1, q2, p1, p2, p3) --> (p1, p2)
# (0,  1 , 2 , 3 , 4 )
# Hence we need to average along axes 0, 1 and 4:
f0 = return_f('dump_f/t=0.000000.bin') # [N_q1, N_q2, N_s, N_p3, N_p2, N_p1]

species = 0

fig = pl.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131)
f0_avged = np.mean(f0, (0, 1, 3))
p2_tmp, p1_tmp = np.meshgrid(p2[species], p1[species])
ax1.contourf(p1_tmp/params.v0, p2_tmp/params.v0, f0_avged[species].transpose(), 100)
ax1.set_aspect('equal')
ax1.set_xlabel('$v_1$')
ax1.set_ylabel('$v_2$')

ax2 = fig.add_subplot(132)
f0_avged = np.mean(f0, (0, 1, 4))
p3_tmp, p1_tmp = np.meshgrid(p3[species], p1[species])
ax2.contourf(p1_tmp/params.v0, p3_tmp/params.v0, f0_avged[species].transpose(), 100)
ax2.set_yticks([])
ax2.set_aspect('equal')
ax2.set_xlabel('$v_1$')
ax2.set_ylabel('$v_3$')

ax3 = fig.add_subplot(133)
f0_avged = np.mean(f0, (0, 1, 5))
p2_tmp, p3_tmp = np.meshgrid(p2[species], p3[species])
ax3.contourf(p3_tmp / params.v0, p2_tmp / params.v0, f0_avged[species], 100)
ax3.set_yticks([])
ax3.set_aspect('equal')
ax3.set_xlabel('$v_3$')
ax3.set_ylabel('$v_2$')

pl.savefig('images/test_f.png')

#for time_index, t0 in enumerate(time_array):
#    
#    # Getting the distribution function at t = 0:
#    # Since we want to visualize variation in (p1, p2), we will be averaging all other quantities:
#    # (q1, q2, p1, p2, p3) --> (p1, p2)
#    # (0,  1 , 2 , 3 , 4 )
#    # Hence we need to average along axes 0, 1 and 4:
#    f = np.mean(return_f_species('dump_f/t=%.3f'%(t0) + '.h5', N_s), (0, 1, 4))
#
#    pl.contourf(p1 / params.v0, p2 / params.v0, abs(f - f0), 100)
#    pl.xlabel(r'$v_x / v_A$')
#    pl.ylabel(r'$v_y / v_A$')
#    pl.title('Time = %.2f'%(t0 / params.t0)+r'$\omega_c^{-1}$')
#    pl.savefig('images_f/%04d'%time_index + '.png')
#    pl.colorbar()
#    pl.clf()
