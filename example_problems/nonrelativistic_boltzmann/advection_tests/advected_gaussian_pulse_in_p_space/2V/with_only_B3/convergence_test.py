import arrayfire as af
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import h5py
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
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
def addition(a, b):
    return(a+b)

def dp_dt(p, t, E1, E2, B3, charge, mass):
    p1 = p[0]
    p2 = p[1]

    dp1_dt = (charge/mass) * (E1 + p2 * B3)
    dp2_dt = (charge/mass) * (E2 - p1 * B3)
    dp_dt  = np.append(dp1_dt, dp2_dt)
    return(dp_dt)

def residual(t_final, E1, E2, B3, charge, mass):
    p1_initial, p2_initial = 2, 2

    t   = np.array([0, t_final])
    sol = odeint(dp_dt, np.array([p1_initial, p2_initial]), t, 
                 args = (E1, E2, B3, charge, mass),
                 rtol = 1e-12, atol = 1e-12
                )

    p1_final, p2_final = sol[-1, 0], sol[-1, 1]
    
    diff_p1  = abs(p1_final - p1_initial)
    diff_p2  = abs(p2_final - p2_initial)
    residual = np.append(diff_p1, diff_p2)
    return(residual)

N = np.array([32, 48, 64, 96, 128])

dt_odeint         = 0.001
t_final_odeint    = 2
time_array_odeint = np.arange(dt_odeint, t_final_odeint + dt_odeint, dt_odeint)

def check_error(params):
    error = np.zeros(N.size)

    for i in range(N.size):
        domain.N_p1 = int(N[i])
        domain.N_p2 = int(N[i])
        # Defining the physical system to be solved:
        system = physical_system(domain,
                                 boundary_conditions,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        nls = nonlinear_solver(system)

        # Time parameters:
        dt = 0.001 * 32/nls.N_p1
        
        # First, we check when the blob returns to (0, 0)
        E1 = nls.fields_solver.cell_centered_EM_fields[0]
        E2 = nls.fields_solver.cell_centered_EM_fields[1]
        B3 = nls.fields_solver.cell_centered_EM_fields[5]

        sol = odeint(dp_dt, np.array([2, 2]), time_array_odeint,
                     args = (af.mean(E1), af.mean(E2), af.mean(B3), 
                             af.sum(params.charge[0]),
                             af.sum(params.mass[0])
                            ),
                     atol = 1e-12, rtol = 1e-12
                    ) 

        dist_from_start = np.sqrt((sol[:, 0] - 2)**2 + (sol[:, 1] - 2)**2)

        pl.plot(sol[:, 0], sol[:, 1])
        pl.show()

        # The time when the distance is minimum apart from the start is the time
        # when the blob returns back to the center:
        # However, this is an approximate solution. To get a more accurate solution, 
        # we provide this guess to our root finder scipy.optimize.root
        t_final_approx = time_array_odeint[np.argmin(dist_from_start[1:])]
        t_final        = root(residual, t_final_approx, 
                              args = (af.mean(E1), af.mean(E2), af.mean(B3), 
                                      params.charge[0],
                                      params.mass[0]
                                     ),
                              method = 'lm', tol = 1e-12
                             ).x

        pl.plot(time_array_odeint, sol[:, 0])
        pl.plot(time_array_odeint, sol[:, 1])
        pl.show()

        time_array  = np.arange(dt, float("{0:.3f}".format(t_final[0])) + dt, dt)

        if(time_array[-1]>t_final):
            time_array = np.delete(time_array, -1)
    
        f_reference = nls.f

        for time_index, t0 in enumerate(time_array):
            nls.strang_timestep(dt)

        error[i] = af.mean(af.abs(  nls.f
                                  - f_reference
                                 )
                          )

    return(error)

minmod_err = check_error(params)
minmod_con = np.polyfit(np.log10(N), np.log10(minmod_err), 1)

print('Error:', minmod_err)
print('Convergence with minmod reconstruction:', minmod_con[0])

pl.loglog(N, minmod_err, '-o', label = 'minmod')
pl.loglog(N, 1e-3/N, '--', color = 'black', label = r'$O(N^{-1})$')
pl.loglog(N, 1e-2/N**2, '-.', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.title('With Upwind-Flux Riemann Solver')
pl.savefig('convergenceplot.png')
