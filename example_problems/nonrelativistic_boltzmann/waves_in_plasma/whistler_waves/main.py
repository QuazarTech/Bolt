import arrayfire as af
import numpy as np
import pylab as pl
import math
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.nonlinear.finite_volume.df_dt_fvm import df_dt_fvm

import domain
import boundary_conditions
import initialize
import params

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 16, 10 #10, 14
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

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1

# Time parameters:
dt_fvm = params.N_cfl * dq1 \
                      / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

dt_fdtd = params.N_cfl * dq1 \
                       / params.c # lightspeed

dt = min(dt_fvm, dt_fdtd)

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

print('Minimum Value of f_e:', af.min(nls.f[:, 0]))
print('Minimum Value of f_i:', af.min(nls.f[:, 1]))

print('Error in density_e:', af.mean(af.abs(nls.compute_moments('density')[:, 0] - 1)))
print('Error in density_i:', af.mean(af.abs(nls.compute_moments('density')[:, 1] - 1)))

v2_bulk = nls.compute_moments('mom_v2_bulk') / nls.compute_moments('density')
v3_bulk = nls.compute_moments('mom_v3_bulk') / nls.compute_moments('density')

v2_bulk_i =   params.amplitude * -8.673617379884035e-17 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * - 0.09260702134169797 * af.sin(params.k_q1 * nls.q1_center)

v2_bulk_e =   params.amplitude * 8.413408858487514e-17 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * - 0.5389501869018833* af.sin(params.k_q1 * nls.q1_center)

v3_bulk_i =   params.amplitude * 0.09260702134169804 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * 3.144186300207963e-18  * af.sin(params.k_q1 * nls.q1_center)

v3_bulk_e =   params.amplitude * 0.5389501869018835 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * -3.885780586188048e-16 * af.sin(params.k_q1 * nls.q1_center)

print('Error in v2_bulk_e:', af.mean(af.abs((v2_bulk[:, 0] - v2_bulk_e) / v2_bulk_e)))
print('Error in v2_bulk_i:', af.mean(af.abs((v2_bulk[:, 1] - v2_bulk_i) / v2_bulk_i)))
print('Error in v3_bulk_e:', af.mean(af.abs((v3_bulk[:, 0] - v3_bulk_e) / v3_bulk_e)))
print('Error in v3_bulk_i:', af.mean(af.abs((v3_bulk[:, 1] - v3_bulk_i) / v3_bulk_i)))

if(params.t_restart == 0):
    time_elapsed = 0
    nls.dump_distribution_function('dump_f/t=0.000')
    nls.dump_moments('dump_moments/t=0.000')
    nls.dump_EM_fields('dump_fields/t=0.000')

else:
    time_elapsed = params.t_restart
    nls.load_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)
    nls.load_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

# Checking that the file writing intervals are greater than dt:
assert(params.dt_dump_f > dt)
assert(params.dt_dump_moments > dt)
assert(params.dt_dump_fields > dt)

while(abs(time_elapsed - params.t_final) > 1e-5):
    
    nls.strang_timestep(dt)
    time_elapsed += dt

    if(params.dt_dump_moments != 0):

        # We step by delta_dt to get the values at dt_dump
        delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
                   * params.dt_dump_moments

        if(delta_dt<dt):
            nls.strang_timestep(delta_dt)
            time_elapsed += delta_dt
            nls.dump_moments('dump_moments/t=' + '%.3f'%time_elapsed)
            nls.dump_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

    if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-5):
        nls.dump_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)

    PETSc.Sys.Print('Computing For Time =', time_elapsed / params.t0, "|t0| units(t0)")
