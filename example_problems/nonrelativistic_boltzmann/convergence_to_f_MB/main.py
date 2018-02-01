import arrayfire as af
import numpy as np
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver

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

@af.broadcast
def f_MB(n, v_bulk, T, q1, v1):
    return(q1**0 * n * (1 / (2 * np.pi * T))**(1 / 2) \
                 * af.exp(-(v1 - v_bulk)**2 / (2 * T))
          )

def set_advection_to_zero(f, t, q1, q2, v1, v2, v3, params):
    return(0 * v1**0, 0 * v2**0)

advection_terms.A_q = set_advection_to_zero
advection_terms.C_q = set_advection_to_zero

# Time parameters:
dt      = 0.001 #* 32 / nls.N_p1
t_final = 2.0

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


time_array  = np.arange(0, t_final + dt, dt)

n_initial  = af.mean(nls.compute_moments('density'))
v1_initial = af.mean(nls.compute_moments('mom_v1_bulk')) / n_initial
T_initial  = (  2 * af.mean(nls.compute_moments('energy'))
               - n_initial * v1_initial**2
             ) / n_initial

f_expected = f_MB(n_initial, v1_initial, T_initial, 
                  nls.q1_center, nls.p1_center
                 )
    
error = np.zeros(time_array.size)
for time_index, t0 in enumerate(time_array):
    # pl.plot(np.array(nls.p1_center).ravel(), np.array(nls.f).ravel())
    # pl.plot(np.array(nls.p1_center).ravel(), np.array(f_expected).ravel(), '--', color = 'black')
    # pl.xlabel(r'$v_1$')
    # pl.ylabel(r'$f$')
    # pl.title('Distribution at Time Index = ' + str(time_index))
    # pl.legend(['Current Distribution', 'Final Expected Distribution'])
    # pl.savefig('images/%04d'%time_index + '.png')
    # pl.clf()
    
    nls.strang_timestep(dt)
    error[time_index] = af.mean(af.abs(nls.f - f_expected))

pl.semilogy(time_array, error)
pl.xlabel('Time')
pl.ylabel('Error')
pl.savefig('logplot.png')
