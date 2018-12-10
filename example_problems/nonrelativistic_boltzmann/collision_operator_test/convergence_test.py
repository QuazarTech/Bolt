import arrayfire as af
import numpy as np
from scipy import integrate
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

@af.broadcast
def f0(v1, n, v_bulk, T):
    return(n * (1 / (2 * np.pi * T))**(1 / 2) \
             * np.exp(-(v1 - v_bulk)**2 / (2 * T))
          )

def BGK(f, t, v1, params):
    dv1 = v1.ravel()[1] - v1.ravel()[0]
    n   = np.sum(f) * dv1

    v_bulk = np.sum(f * v1) * dv1 / n

    T = (  np.sum(f * v1**2) * dv1
         - n * v_bulk**2
        ) / n

    f_MB = f0(v1, n, v_bulk, T)
    tau  = params.tau(0, 0, 0, 0, 0)

    C_f = -(f - f_MB) / tau
    return(C_f)

def set_advection_to_zero(t, q1, q2, v1, v2, v3, params):
    return(0 * v1**0, 0 * v2**0)

advection_terms.A_q = set_advection_to_zero
advection_terms.C_q = set_advection_to_zero

system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

nls       = nonlinear_solver(system)
f_initial = nls.f.copy()

# Time parameters:
dt_samples = 1e-4 / 2**np.arange(5)
t_final    = 0.01
N_t        = t_final / dt_samples # Number of timesteps

sol = integrate.odeint(BGK, np.array(nls.f), np.array([0, t_final]),
                       args = (np.array(nls.p1_center), params),
                       rtol = 1e-20, atol = 5e-14
                      )

f_reference = sol[-1].ravel()

error = np.zeros(dt_samples.size)

for i in range(dt_samples.size):

    # Declaring a linear system object which will evolve the defined physical system:
    dt  = dt_samples[i]
    
    time_array = np.arange(0, t_final + dt, dt)

    for time_index, t0 in enumerate(time_array[1:]):
        nls.strang_timestep(dt)

    error[i] = np.mean(abs(np.array(nls.f) - f_reference))
    # Setting back to initial value for next iteration:
    nls.f    = f_initial

print(N_t)
print(error)

print('L1 norm of error:', error)
print('Order of convergence:', np.polyfit(np.log10(N_t), np.log10(error), 1)[0])

pl.loglog(N_t, error, '-o', label = 'Numerical')
pl.loglog(N_t, error[0]*N_t[0]**2/N_t**2, '--', color = 'black', label = r'$\mathcal{O}(N_t^{-2})$')
pl.xlabel(r'$N_t$')
pl.ylabel('Error')
pl.legend(fontsize = 28, framealpha = 0)
pl.savefig('convergenceplot.png', bbox_inches = 'tight')
