import numpy as np 
import h5py
import matplotlib as mpl
mpl.use('agg')
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

# Checking the errors
def check_convergence():
    N     = 2**np.arange(5, 8)
    error = np.zeros(N.size)
    
    for i in range(N.size):

        h5f   = h5py.File('dump_files/nlsf_' + str(N[i]) + '.h5')
        nls_f = h5f['distribution_function'][:]
        h5f.close()    

        h5f  = h5py.File('dump_files/lsf_' + str(N[i]) + '.h5')
        ls_f = h5f['distribution_function'][:]
        h5f.close()

        error[i] = np.mean(abs(nls_f - ls_f))

    print(error)
    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    print(poly)

    pl.loglog(N, error, 'o-', label = 'Numerical')
    pl.loglog(N, error[0]*32**2/N**2, '--', color = 'black', 
              label = r'$O(N^{-2})$'
             )
    pl.legend(loc = 'best')
    pl.ylabel('Error')
    pl.xlabel('$N$')
    pl.savefig('convergence_plot.png')

    assert(abs(poly[0] + 2)<0.25)
