import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
from scipy.ndimage.filters import gaussian_filter

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver
from bolt.lib.utils.broadcasted_primitive_operations import multiply

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

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

def lowpass_filter(f):
    f_hat = af.fft(f)
    dp1   = (domain.p1_end - domain.p1_start) / domain.N_p1
    k_v   = af.tile(af.to_array(np.fft.fftfreq(domain.N_p1, dp1)), 
                    1, 1, f.shape[2], f.shape[3]
                   )
    
    # Applying the filter:
    f_hat_filtered = 0.5 * (f_hat * (  af.tanh((k_v + 0.9 * af.max(k_v)) / 0.5)
                                     - af.tanh((k_v + 0.9 * af.min(k_v)) / 0.5)
                                    )
                           )

    f_hat = af.select(af.abs(k_v) < 0.8 * af.max(k_v), f_hat, f_hat_filtered)
    f = af.real(af.ifft(f_hat))
    return(f) 

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

N_g = system.N_ghost


# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
ls  = linear_solver(system)

# Time parameters:
dt      = 0.0005
t_final = 1.0

time_array  = np.arange(0, t_final + dt, dt)

rho_data_nls = np.zeros(time_array.size)
rho_data_ls  = np.zeros(time_array.size)

# Storing data at time t = 0:
n_nls           = nls.compute_moments('density')
rho_data_nls[0] = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])
n_ls           = ls.compute_moments('density')
rho_data_ls[0] = af.max(n_ls)

f_initial = nls.f.copy()

for time_index, t0 in enumerate(time_array[1:]):
    #nls.dump_distribution_function('dump/%04d'%time_index)
    nls.strang_timestep(dt)
    # nls.f = lowpass_filter(nls.f)
    # nls.f = af.to_array(gaussian_filter(np.array(nls.f), (0.2, 0, 0, 0)))
    ls.RK4_timestep(dt)
    
#    if(time_index % 25 == 0):
#        nls.f = lowpass_filter(nls.f)

        
        # f_hat = af.fft2(nls.f)
    #     # f_hat[16] = 0
    #     # nls.f = af.abs(af.ifft2(f_hat, scale = 1) / 32)
    #     nls.f = af.to_array(gaussian_filter(np.array(nls.f), (0.5, 0, 0, 0)))
    #     # delta_f = nls.f - f_initial
    #     # filtered_delta_f = af.to_array(gaussian_filter(np.array(delta_f), (1, 0, 0, 0)))
    #     # nls.f = f_initial + filtered_delta_f

    n_nls                         = nls.compute_moments('density')
    rho_data_nls[time_index + 1]  = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])
    
    n_ls                        = ls.compute_moments('density')
    rho_data_ls[time_index + 1] = af.max(n_ls) 

# h5f = h5py.File('sigma_point5.h5', 'w')
# h5f.create_dataset('n', data = rho_data_nls)
# h5f.create_dataset('time', data = time_array)
# h5f.close()

# h5f = h5py.File('unfiltered.h5', 'r')
# n1 = h5f['n'][:]
# h5f.close()

# h5f = h5py.File('sigma_2.h5', 'r')
# n2 = h5f['n'][:]
# h5f.close()

# h5f = h5py.File('sigma_1.h5', 'r')
# n3 = h5f['n'][:]
# h5f.close()

# pl.plot(time_array, n1, label='Unfiltered')
pl.plot(time_array, rho_data_nls, label = 'Nonlinear Solver')
# pl.plot(time_array, n3, label=r'$\sigma=1$')
# pl.plot(time_array, n2, label=r'$\sigma=2$')
pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($n$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('n.png')
pl.savefig('n.svg')
pl.clf()

pl.semilogy(time_array, rho_data_nls-1, label = 'Nonlinear Solver')
# pl.plot(time_array, n3, label=r'$\sigma=1$')
# pl.plot(time_array, n2, label=r'$\sigma=2$')
pl.semilogy(time_array, rho_data_ls-1, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($n$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('n_semilogy.png')
pl.savefig('n_semilogy.svg')
pl.clf()
