import h5py
import pylab as pl 
import numpy as np
# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
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

# # Importing density evolution as given by the CK code:
h5f = h5py.File('ck_density_data.h5', 'r')
amplitude_ck = h5f['density_amplitude'][:]
time_ck      = h5f['time'][:]
h5f.close()

# Importing density evolutions as given by the LT code:
h5f = h5py.File('lt_density_data.h5', 'r')
amplitude_lt = h5f['density_amplitude'][:]
time_lt      = h5f['time'][:]
h5f.close()

# # Plotting:
# h5f  = h5py.File('ck_distribution_function0.h5', 'r')
# f_ck = h5f['distribution_function'][:]
# h5f.close()

# h5f  = h5py.File('ck_distribution_function.h5', 'r')
# f_ck1 = h5f['distribution_function'][:]
# h5f.close()

# h5f  = h5py.File('lt_distribution_function.h5', 'r')
# f_lt = h5f['distribution_function'][:]
# h5f.close()

# f_ck = np.swapaxes(f_ck, 0, 1).reshape(f_lt.shape[0], f_lt.shape[1], f_lt.shape[4], f_lt.shape[3], f_lt.shape[2])
# f_ck = np.swapaxes(f_ck, 4, 2)
# f_ck1 = np.swapaxes(f_ck1, 0, 1).reshape(f_lt.shape[0], f_lt.shape[1], f_lt.shape[4], f_lt.shape[3], f_lt.shape[2])
# f_ck1 = np.swapaxes(f_ck1, 4, 2)

pl.plot(time_ck[:15], amplitude_ck[:15], label = 'CK')
# pl.plot(time_lt, amplitude_lt, '--', color = 'black', label = 'LT')

# x = (0.5 + np.arange(32))*(1/32)
# v = -9 + (0.5 + np.arange(64))*(18/64)

# x, v = np.meshgrid(x, v)
# f_ck = np.swapaxes(f_ck, 0, 3)
# f_ck1 = np.swapaxes(f_ck1, 0, 3)

pl.xlabel('Time')
pl.ylabel(r'$MIN(\delta \rho(x))$')
pl.legend()
# # f_test = np.zeros_like(f_ck[:, :, 0, 0, 0])
# # f_test = np.where(f_ck[:, :, 0, 0, 0]<0, 1, f_test)

# pl.contourf(x, v, abs(f_ck[:, :, 0, 0, 0] - f_ck1[:, :, 0, 0, 0]), 100)
# pl.colorbar()
# pl.xlabel(r'$x$')
# pl.ylabel(r'$v$')
# pl.title('Time = 0.08')
pl.savefig('plot.png')