import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
pl.rcParams['figure.dpi']      = 300
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3
N_g  = domain.N_ghost

dq1 = (domain.q1_end - domain.q1_start) / N_q1
dq2 = (domain.q2_end - domain.q2_start) / N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

def return_f_species(file_name, N_s = 0):
    """
    Returns the distribution function in the desired format of 
    (q1, q2, p1, p2, p3), when the file that is written by the 
    dump_distribution_function is passed as the argument

    Parameters
    ----------

    file_name : str
                Pass the name of the file that needs to be read as
                a string. This should be the file that is written by
                dump_distribution_function

    N_s: int
         Pass the species that is in consideration. For instance when
         the run is carried out for 2 species, passing N_s = 0 returns
         the distribution function for the first species while N_s = 1
         returns the distribution function of the second species.
    """

    # When written using the routine dump_distribution_function, 
    # distribution function gets written in the format (q2, q1, p1 * p2 * p3 * Ns)
    h5f = h5py.File(file_name, 'r')
    # Making this (q2, q1, p1 * p2 * p3 * Ns) --> (q1, q2, p1 * p2 * p3 * Ns)
    f   = np.swapaxes(h5f['distribution_function'][:], 0, 1)
    h5f.close()

    # Finding the distribution function for the species in consideration:
    f = f[:, :, N_s * N_p1 * N_p2 * N_p3:(N_s + 1) * N_p1 * N_p2 * N_p3]
    # Reshaping this from (q1, q2, p1 * p2 * p3) --> (q1, q2, p1, p2, p3)
    f = f.reshape(N_q1, N_q2, N_p1, N_p2, N_p3)

    return f

# Set this to 0 for electrons, 1 for ions:
N_s = 0

dp1 = (domain.p1_end[N_s] - domain.p1_start[N_s]) / N_p1
dp2 = (domain.p2_end[N_s] - domain.p2_start[N_s]) / N_p2
dp3 = (domain.p3_end[N_s] - domain.p3_start[N_s]) / N_p3

p1 = domain.p1_start[N_s] + (0.5 + np.arange(N_p1)) * dp1
p2 = domain.p2_start[N_s] + (0.5 + np.arange(N_p2)) * dp2
p3 = domain.p3_start[N_s] + (0.5 + np.arange(N_p3)) * dp3

# We want to see the distribution function variation as a function of (p1, p2)
# Hence we take a meshgrid of (p1, p2)
p1, p2 = np.meshgrid(p1, p2)

# Declaration of the time array:
time_array = np.arange(0, params.t_final + params.dt_dump_f, 
                       params.dt_dump_f
                      )

# Getting the distribution function at t = 0:
# Since we want to visualize variation in (p1, p2), we will be averaging all other quantities:
# (q1, q2, p1, p2, p3) --> (p1, p2)
# (0,  1 , 2 , 3 , 4 )
# Hence we need to average along axes 0, 1 and 4:
f0 = np.mean(return_f_species('dump_f/t=0.000.h5', N_s), (0, 1, 4))

for time_index, t0 in enumerate(time_array):
    
    # Getting the distribution function at t = 0:
    # Since we want to visualize variation in (p1, p2), we will be averaging all other quantities:
    # (q1, q2, p1, p2, p3) --> (p1, p2)
    # (0,  1 , 2 , 3 , 4 )
    # Hence we need to average along axes 0, 1 and 4:
    f = np.mean(return_f_species('dump_f/t=%.3f'%(t0) + '.h5', N_s), (0, 1, 4))

    pl.contourf(p1 / params.v0, p2 / params.v0, abs(f - f0), 100)
    pl.xlabel(r'$v_x / v_A$')
    pl.ylabel(r'$v_y / v_A$')
    pl.title('Time = %.2f'%(t0 / params.t0)+r'$\omega_c^{-1}$')
    pl.savefig('images_f/%04d'%time_index + '.png')
    pl.colorbar()
    pl.clf()
