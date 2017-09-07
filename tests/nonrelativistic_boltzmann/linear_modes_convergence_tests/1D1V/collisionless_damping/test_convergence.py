import arrayfire as af
import numpy as np
import h5py
import pylab as pl

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver

from bolt.lib.linear_solver.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms

import bolt.src.nonrelativistic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize'] = 12, 7.5
pl.rcParams['figure.dpi'] = 300
pl.rcParams['image.cmap'] = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family'] = 'serif'
pl.rcParams['font.weight'] = 'bold'
pl.rcParams['font.size'] = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex'] = True
pl.rcParams['axes.linewidth'] = 1.5
pl.rcParams['axes.titlesize'] = 'medium'
pl.rcParams['axes.labelsize'] = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad'] = 8
pl.rcParams['xtick.minor.pad'] = 8
pl.rcParams['xtick.color'] = 'k'
pl.rcParams['xtick.labelsize'] = 'medium'
pl.rcParams['xtick.direction'] = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad'] = 8
pl.rcParams['ytick.minor.pad'] = 8
pl.rcParams['ytick.color'] = 'k'
pl.rcParams['ytick.labelsize'] = 'medium'
pl.rcParams['ytick.direction'] = 'in'

# Time parameters:
t_final = 0.1
N       = 2**np.arange(5, 10)

def run_cases():
    # Running the setup for all resolutions:
    for i in range(N.size):
        af.device_gc()
        domain.N_q1 = int(N[i])
        dt          = 0.001/(2**i)

        # Defining the physical system to be solved:
        system = physical_system(domain,\
                                 boundary_conditions,\
                                 params,\
                                 initialize,\
                                 advection_terms,\
                                 collision_operator.BGK,\
                                 moment_defs
                                )

        # Declaring a linear system object which will 
        # evolve the defined physical system:
        nls = nonlinear_solver(system)
        ls  = linear_solver(system)

        time_array = np.arange(dt, t_final + dt, dt)

        for time_index, t0 in enumerate(time_array):

            nls.strang_timestep(dt)
            ls.RK2_timestep(dt)

        nls.dump_distribution_function('dump_files/nlsf_' + str(N[i]))
        ls.dump_distribution_function('dump_files/lsf_' + str(N[i]))

        nls.dump_moments('dump_files/nlsm_' + str(N[i]))
        ls.dump_moments('dump_files/lsm_' + str(N[i]))

# Checking the errors
def test_convergence():
    
    error = np.zeros(N.size)
    run_cases()
    
    for i in range(N.size):
        h5f   = h5py.File('dump_files/nlsm_' + str(N[i]) + '.h5')
        nls_f = h5f['moments'][:]
        h5f.close()    

        h5f  = h5py.File('dump_files/lsm_' + str(N[i]) + '.h5')
        ls_f = h5f['moments'][:]
        h5f.close()

        error[i] = np.mean(abs(nls_f - ls_f))

    # pl.loglog(N, error, 'o-', label = 'Numerical')
    # pl.loglog(N, error[0]*32**2/N**2, '--', color = 'black', 
    #           label = r'$O(N^{-2})$')
    # pl.legend(loc = 'best')
    # pl.ylabel('Error')
    # pl.xlabel('$N$')
    # pl.savefig('convergence_plot.png')

    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    assert(abs(poly[0] + 2)<0.25)
