import h5py
import pylab as pl 

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

# Importing density evolution as given by the CK code:
h5f = h5py.File('ck_density_data.h5', 'r')
amplitude_ck = h5f['density_amplitude'][:] + 1
time_ck      = h5f['time'][:]
h5f.close()

# Importing density evolutions as given by the LT code:
h5f = h5py.File('lt_density_data.h5', 'r')
amplitude_lt = h5f['density_amplitude'][:]
time_lt      = h5f['time'][:]
h5f.close()

# Plotting:
pl.plot(time_ck, amplitude_ck, label = 'CK')
# pl.semilogy(time_lt, amplitude_lt, '--', color = 'black', label = 'LT')
pl.xlabel('Time')
pl.ylabel(r'$MAX(\delta \rho(x))$')
pl.legend()
pl.savefig('plot.png')