import numpy as np
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import h5py

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

h5f = h5py.File('collisionless/data.h5', 'r')
time_array_1  = h5f['time'][:]
E_ls_tau_inf  = h5f['electrical_energy_ls'][:]
E_nls_tau_inf = h5f['electrical_energy_nls'][:]
h5f.close()

h5f = h5py.File('tau_0.01/data.h5', 'r')
time_array_2         = h5f['time'][:]
E_ls_tau_1e_minus_2  = h5f['electrical_energy_ls'][:]
E_nls_tau_1e_minus_2 = h5f['electrical_energy_nls'][:]
h5f.close()

#h5f = h5py.File('tau_0.001/data.h5', 'r')
#time_array_3         = h5f['time'][:]
#E_ls_tau_1e_minus_3  = h5f['electrical_energy_ls'][:]
#E_nls_tau_1e_minus_3 = h5f['electrical_energy_nls'][:]
#h5f.close()

h5f = h5py.File('tau_0/data.h5', 'r')
time_array_4 = h5f['time'][:]
E_ls_tau_0   = h5f['electrical_energy_ls'][:]
E_nls_tau_0  = h5f['electrical_energy_nls'][:]
h5f.close()

h5f = h5py.File('tau_1/data.h5', 'r')
time_array_5 = h5f['time'][:]
E_ls_tau_1   = h5f['electrical_energy_ls'][:]
E_nls_tau_1  = h5f['electrical_energy_nls'][:]
h5f.close()

h5f = h5py.File('tau_10/data.h5', 'r')
time_array_6 = h5f['time'][:]
E_ls_tau_10  = h5f['electrical_energy_ls'][:]
E_nls_tau_10 = h5f['electrical_energy_nls'][:]
h5f.close()

h5f = h5py.File('tau_100/data.h5', 'r')
time_array_7  = h5f['time'][:]
E_ls_tau_100  = h5f['electrical_energy_ls'][:]
E_nls_tau_100 = h5f['electrical_energy_nls'][:]
h5f.close()

pl.plot(time_array_1, E_nls_tau_inf, color = 'blue', label=r'$\tau=\inf$')
pl.plot(time_array_1, E_ls_tau_inf, '--', color = 'blue', label = r'$\tau=\inf$')
pl.plot(time_array_7, E_nls_tau_100, color = 'orangered', label=r'$\tau=100$')
pl.plot(time_array_7, E_ls_tau_100, '--', color = 'orangered', label = r'$\tau=100$')
pl.plot(time_array_6, E_nls_tau_10, color = 'maroon', label=r'$\tau=10$')
pl.plot(time_array_6, E_ls_tau_10, '--', color = 'maroon', label = r'$\tau=10$')
pl.plot(time_array_5, E_nls_tau_1, color = 'olive', label=r'$\tau=1$')
pl.plot(time_array_5, E_ls_tau_1, '--', color = 'olive', label = r'$\tau=1$')
pl.plot(time_array_2, E_nls_tau_1e_minus_2, color = 'green', label=r'$\tau=0.01$')
pl.plot(time_array_2, E_ls_tau_1e_minus_2, '--', color = 'green', label = r'$\tau=0.01$')
#pl.plot(time_array_3, E_nls_tau_1e_minus_3, color = 'red', label=r'$\tau=0.001$')
#pl.plot(time_array_3, E_ls_tau_1e_minus_3, '--', color = 'red', label = r'$\tau=0.001$')
pl.plot(time_array_4, E_nls_tau_0, color = 'cyan', label=r'$\tau=0$')
pl.plot(time_array_4, E_ls_tau_0, '--', color = 'cyan', label = r'$\tau=0$')
pl.ylabel(r'SUM($|E|^2$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('linearplot.png')
pl.clf()

pl.semilogy(time_array_1, E_nls_tau_inf, color = 'blue', label=r'$\tau=\infty$')
pl.semilogy(time_array_1, E_ls_tau_inf, '--', color = 'blue', label = r'$\tau=\infty$')
pl.semilogy(time_array_7, E_nls_tau_100, color = 'orangered', label=r'$\tau=100$')
pl.semilogy(time_array_7, E_ls_tau_100, '--', color = 'orangered', label = r'$\tau=100$')
pl.semilogy(time_array_6, E_nls_tau_10, color = 'maroon', label=r'$\tau=10$')
pl.semilogy(time_array_6, E_ls_tau_10, '--', color = 'maroon', label = r'$\tau=10$')
pl.semilogy(time_array_5, E_nls_tau_1, color = 'olive', label=r'$\tau=1$')
pl.semilogy(time_array_5, E_ls_tau_1, '--', color = 'olive', label = r'$\tau=1$')
pl.semilogy(time_array_2, E_nls_tau_1e_minus_2, color = 'green', label=r'$\tau=0.01$')
pl.semilogy(time_array_2, E_ls_tau_1e_minus_2, '--', color = 'green', label = r'$\tau=0.01$')
#pl.semilogy(time_array_3, E_nls_tau_1e_minus_3, color = 'red', label=r'$\tau=0.001$')
#pl.semilogy(time_array_3, E_ls_tau_1e_minus_3, '--', color = 'red', label = r'$\tau=0.001$')
pl.semilogy(time_array_4, E_nls_tau_0, color = 'cyan', label=r'$\tau=0$')
pl.semilogy(time_array_4, E_ls_tau_0, '--', color = 'cyan', label = r'$\tau=0$')
pl.ylabel(r'SUM($|E|^2$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('semilogyplot.png')
pl.clf()
