import arrayfire as af
import numpy as np
import math
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import initialize
import params

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

import matplotlib
matplotlib.use('agg')
import pylab as pl

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 9 #10, 14
pl.rcParams['figure.dpi']      = 80
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

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost

pl.plot(nls.p1_center[:, 0], af.flat(nls.f[:, 0, 16, 0]), label = r'Species-1($T = T_0$)')
pl.plot(nls.p1_center[:, 1], af.flat(nls.f[:, 1, 16, 0]), label = r'Species-2($T = 0.1 T_0$)')
pl.plot(nls.p1_center[:, 2], af.flat(nls.f[:, 2, 16, 0]), label = r'Species-3($T = 0.01 T_0$)')
pl.xlabel('$v$')
pl.ylabel('$f$')
pl.legend()
pl.savefig('initial_dist.png')
pl.clf()

# Time parameters:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

# if(params.t_restart == 0):
#     time_elapsed = 0
#     nls.dump_distribution_function('dump_f/t=0.000')
#     nls.dump_moments('dump_moments/t=0.000')
#     nls.dump_EM_fields('dump_fields/t=0.000')

# else:
#     time_elapsed = params.t_restart
#     nls.load_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)
#     nls.load_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

# Checking that the file writing intervals are greater than dt:
# assert(params.dt_dump_f > dt)
# assert(params.dt_dump_moments > dt)
# assert(params.dt_dump_fields > dt)

time_array = []
data1      = [] 
data2      = [] 
data3      = [] 

time_elapsed = 0

while(abs(time_elapsed - params.t_final) > 1e-12):

    time_array.append(time_elapsed)
    data1.append(af.max(nls.compute_moments('density')[:, 0]))
    data2.append(af.max(nls.compute_moments('density')[:, 1]))
    data3.append(af.max(nls.compute_moments('density')[:, 2]))
    
    nls.strang_timestep(dt)
    time_elapsed += dt

    # if(params.dt_dump_moments != 0):

    #     # We step by delta_dt to get the values at dt_dump
    #     delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
    #                * params.dt_dump_moments

    #     if(delta_dt<dt):
    #         nls.strang_timestep(delta_dt)
    #         time_elapsed += delta_dt
    #         nls.dump_moments('dump_moments/t=' + '%.3f'%time_elapsed)
    #         nls.dump_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

    # if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-12):
    #     nls.dump_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)

    PETSc.Sys.Print('Computing For Time =', time_elapsed / params.t0, "|t0| units(t0)")

time_array = np.array(time_array)
data1      = np.array(data1)
data2      = np.array(data2)
data3      = np.array(data3)

pl.plot(time_array / params.t0, data1, label = r'Species-1($T = T_0$)')
pl.plot(time_array / params.t0, data2, label = r'Species-2($T = 0.1 T_0$)')
pl.plot(time_array / params.t0, data3, label = r'Species-3($T = 0.01 T_0$)')
pl.xlabel(r'Time($\frac{L_0}{c_s}$)')
pl.ylabel(r'$\max(n)$')
pl.legend()
pl.savefig('plot.png')
