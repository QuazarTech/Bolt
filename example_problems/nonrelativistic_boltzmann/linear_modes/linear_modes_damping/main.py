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
pl.rcParams['figure.figsize']  = 9, 4
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

def lowpass_filter(f):
    f_hat = af.fft(f)
    dp1   = (domain.p1_end[0] - domain.p1_start[0]) / domain.N_p1
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
nls  = nonlinear_solver(system)
nls2 = nonlinear_solver(system)
ls   = linear_solver(system)

# Time parameters:
# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

rho_data_nls  = np.zeros(time_array.size)
rho_data_nls2 = np.zeros(time_array.size)
rho_data_ls   = np.zeros(time_array.size)

# Storing data at time t = 0:
n_nls           = nls.compute_moments('density')
rho_data_nls[0] = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])

n_nls2           = nls2.compute_moments('density')
rho_data_nls2[0] = af.max(n_nls2)

n_ls           = ls.compute_moments('density')
rho_data_ls[0] = af.max(n_ls)

# Printing the error in moment computation:
n = params.n_background + params.alpha * af.cos(0.5 * nls.q1_center)
print('Error in density computation = ', af.sum(af.abs(n_nls - n)) / n.elements())

for time_index, t0 in enumerate(time_array[1:]):
    print(t0)

    if(time_index % 25 == 0):
        nls.f = lowpass_filter(nls.f)

    nls.strang_timestep(dt)
    nls2.strang_timestep(dt)
    ls.RK4_timestep(dt)

    n_nls                      = nls.compute_moments('density')
    rho_data_nls[time_index+1] = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])

    n_nls2                        = nls2.compute_moments('density')
    rho_data_nls2[time_index + 1] = af.max(n_nls2[:, :, N_g:-N_g, N_g:-N_g]) 

    n_ls                        = ls.compute_moments('density')
    rho_data_ls[time_index + 1] = af.max(n_ls)

h5f = h5py.File('dump/data_Nx_'+ str(domain.N_q1) + '_Nv_' + str(domain.N_p1) + '.h5', 'w')
h5f.create_dataset('n_nls', data = rho_data_nls)
h5f.create_dataset('n_nls2', data = rho_data_nls2)
h5f.create_dataset('n_ls', data = rho_data_ls)
h5f.create_dataset('time', data = time_array)
h5f.close()

pl.plot(time_array, rho_data_nls2, label = 'Without Filter')
pl.plot(time_array, rho_data_nls, '--', color = 'red', label = 'With Filter')
pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Theory')
pl.ylabel('$n$')
pl.xlabel('Time')
pl.legend(fontsize = 25)
pl.savefig('n.png', bbox_inches = 'tight')
pl.clf()

pl.semilogy(time_array, rho_data_nls2-1, label = 'Without Filter')
pl.semilogy(time_array, rho_data_nls-1, '--', color = 'red', label = 'With Filter')
pl.plot(time_array, rho_data_ls-1, '--', color = 'black', label = 'Linear Theory')
pl.ylabel(r'$\delta n$')
pl.xlabel('Time')
pl.legend(fontsize = 25)
pl.savefig('n_semilogy.png', bbox_inches = 'tight')
pl.clf()
