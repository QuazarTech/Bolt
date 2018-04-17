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
dt      = 0.001
t_final = 0.5

time_array  = np.arange(0, t_final + dt, dt)

rho_data_nls = np.zeros(time_array.size)
rho_data_ls  = np.zeros(time_array.size)

# Storing data at time t = 0:
n_nls           = nls.compute_moments('density')
rho_data_nls[0] = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])
n_ls           = ls.compute_moments('density')
rho_data_ls[0] = af.max(n_ls)

print(rho_data_nls[0])
print(rho_data_ls[0])

f_initial = nls.f.copy()

nls.data = np.zeros(time_array.size)
nls.count = 0

for time_index, t0 in enumerate(time_array[1:]):

    nls.strang_timestep(dt)
    ls.RK4_timestep(dt)
    print(t0)

    # nls.f = af.to_array(gaussian_filter(np.array(nls.f), (0.2, 0, 0, 0)))
    # if(time_index % 25 == 0):
    #     nls.f = lowpass_filter(nls.f)

    n_nls                      = nls.compute_moments('density')
    rho_data_nls[time_index+1] = af.max(n_nls[:, :, N_g:-N_g, N_g:-N_g])

    rho_n = -10 * nls.compute_moments('density')
    rho_n = af.sum(rho_n, 1)

    # nls.fields_solver.check_maxwells_contraint_equations(rho_n)

    # nls.strang_timestep(dt)

    rho_n_plus_one       = -10 * nls.compute_moments('density')
    rho_n_plus_one       = af.sum(rho_n_plus_one, 1)

    # divE      = nls.fields_solver.compute_divE()
    drho_dt = (rho_n_plus_one - rho_n) / dt
    #J1 = nls.fields_solver.J1
    #J2 = nls.fields_solver.J2

    #J1_plus_q1 = af.shift(nls.fields_solver.J1, 0, 0, -1)
    #J2_plus_q2 = af.shift(nls.fields_solver.J2, 0, 0, 0, -1)

    #divJ = (J1_plus_q1 - J1) / nls.dq1 + 0*(J2_plus_q2 - J2) / nls.dq2

    # print(af.mean(af.abs(drho_dt + divJ)[:, :, 3:-3, 3:-3]))

    n_ls                        = ls.compute_moments('density')
    rho_data_ls[time_index + 1] = af.max(n_ls) 

pl.plot(time_array, rho_data_nls, label = 'Nonlinear Solver')
# pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel('Error')
pl.xlabel('Time')
pl.legend()
pl.savefig('n.png')
pl.savefig('n.svg')
pl.clf()

pl.semilogy(time_array, rho_data_nls-1, label = 'Nonlinear Solver')
# pl.semilogy(time_array, rho_data_ls-1, '--', color = 'black', label = 'Linear Solver')
pl.ylabel('Error')
pl.xlabel('Time')
pl.legend()
pl.savefig('n_semilogy.png')
pl.savefig('n_semilogy.svg')
pl.clf()

