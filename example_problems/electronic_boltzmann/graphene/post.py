import numpy as np
import h5py
import domain
import params
import matplotlib
matplotlib.use('agg')
import pylab as pl

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
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

#time = np.arange(0, params.t_final + params.dt_dump_moments, 
#                 params.dt_dump_moments
#                )

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2, q1 = np.meshgrid(q2, q1)

#dump_index = 000424
#h5f  = h5py.File('dumps/density_' + str(dump_index) + '.h5', 'r')
h5f  = h5py.File('dumps/density_000030.h5', 'r')
moments = np.swapaxes(h5f['moments'][:], 0, 1)
h5f.close()

density = moments[:, :]
pl.contourf(q1, q2, density, 100)
#pl.title('Time = ' + "%.2f"%(t0))
pl.axes().set_aspect('equal')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.colorbar()
pl.savefig('images/density' + '.png')
#pl.savefig('images/%04d'%time_index + '.png')
pl.clf()


h5f  = h5py.File('dumps/fields_000030.h5', 'r')
fields = np.swapaxes(h5f['fields'][:], 0, 1)
h5f.close()

E1 = fields[:, :, 0]
E2 = fields[:, :, 1]

pl.figure(figsize=(20, 7.5))
pl.subplot(121)
pl.contourf(q1, q2, E1, 100, cmap='bwr')
pl.title('$E_1$')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.colorbar()
pl.gca().set_aspect('equal')

pl.subplot(122)
pl.contourf(q1, q2, E2, 100, cmap='bwr')
pl.title('$E_2$')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.colorbar()
pl.gca().set_aspect('equal')
pl.savefig('images/E_fields' + '.png')
pl.clf()

#for time_index, t0 in enumerate(time):
#    
#    h5f  = h5py.File('dumps/density_' + str(time_index) + '.h5', 'r')
#    moments = np.swapaxes(h5f['moments'][:], 0, 1)
#    h5f.close()
#    
#    n = moments[:, :, 0]
#
#    #pl.contourf(q1, q2, n, np.linspace(0.8, 2.2, 500))
#    pl.contourf(q1, q2, n, 100)
#    pl.title('Time = ' + "%.2f"%(t0))
#    pl.axes().set_aspect('equal')
#    pl.xlabel(r'$x$')
#    pl.ylabel(r'$y$')
#    pl.colorbar()
#    pl.savefig('images/%04d'%time_index + '.png')
#    pl.clf()
