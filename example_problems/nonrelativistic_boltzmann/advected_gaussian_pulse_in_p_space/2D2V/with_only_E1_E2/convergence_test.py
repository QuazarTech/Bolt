import arrayfire as af
import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver
from bolt.lib.linear_solver.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 20, 10
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

N     = 2**np.arange(5, 8) #np.array([32, 48, 64, 96])
error = np.zeros(3)

for i in range(N.size):
    af.device_gc()
    domain.N_p1 = int(N[i])
    domain.N_p2 = int(N[i])

    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moment_defs
                            )

    N_g_q = system.N_ghost_q


    # Declaring a linear system object which will evolve the defined physical system:
    nls = nonlinear_solver(system)
    #ls  = linear_solver(system)

    # print("N_q1 =", nls.N_q1, ", N_q2 =", nls.N_q2, ", N_p1 =", nls.N_p1, ", N_p2 =", nls.N_p2)

    # params2 = params

    # params2.solver_method_in_q = 'ASL'
    # params2.solver_method_in_p = 'ASL'

    # domain2 = domain

    # domain2.N_p1 = int(2048)
    # domain2.N_p2 = int(2048)

    # system2 = physical_system(domain2,
    #                           boundary_conditions,
    #                           params2,
    #                           initialize,
    #                           advection_terms,
    #                           collision_operator.BGK,
    #                           moment_defs
    #                          )

    # nls2 = nonlinear_solver(system2)

    p1 = np.array(af.moddims(nls.p1_center, nls.N_p1, nls.N_p2))
    p2 = np.array(af.moddims(nls.p2_center, nls.N_p1, nls.N_p2))

    # p1prime = np.array(af.moddims(nls2.p1_center, nls2.N_p1, nls2.N_p2))
    # p2prime = np.array(af.moddims(nls2.p2_center, nls2.N_p1, nls2.N_p2))

    # f_at_desired_q_initial = af.moddims(nls.f[:, N_g, N_g + nls.N_q2/2],
    #                             nls.N_p1, nls.N_p2
    #                            )

    # pl.contourf(p1, p2, np.array(f_at_desired_q_initial), 100, cmap='bwr')
    # pl.gca().set_aspect('equal')


    # In[8]:


    # Time parameters:
    dt      = 0.001 * 32/nls.N_p1
    t_final = 0.1

    time_array  = np.arange(0, t_final + dt, dt)
    
    if(time_array[-1]>t_final):
        time_array = np.delete(time_array, -1)

    f_initial = nls.f.copy()

    maxf = af.max(nls.f) + 0.02
    minf = af.min(nls.f) - 0.02

    # f_initial = 0.5 * ls.N_q1 * ls.N_q2 * af.ifft2(ls.Y[:, :, :, 0]) 
    from initialize import initialize_f

    for time_index, t0 in enumerate(time_array[1:]):
        print("time_index = ", time_index, " of ", time_array.size-2, " t = ", t0)
        nls.lie_timestep(dt)
        # nls2.lie_timestep(t0)
        #ls.RK4_timestep(dt)
        
        #f = 0.5 * ls.N_q1 * ls.N_q2 * af.ifft2(ls.Y[:, :, :, 0])

    # nls.lie_timestep(t_final)

        # f_at_desired_q1 = af.moddims(nls.f[:, 1, 1],
        #                              nls.N_p1, nls.N_p2
        #                             )
    
        E1 = nls.cell_centered_EM_fields_at_n[0]
        E2 = nls.cell_centered_EM_fields_at_n[1]
        E3 = nls.cell_centered_EM_fields_at_n[2]

        B1 = nls.cell_centered_EM_fields_at_n[3]
        B2 = nls.cell_centered_EM_fields_at_n[4]
        B3 = nls.cell_centered_EM_fields_at_n[5]

        (A_p1, A_p2, A_p3) = af.broadcast(nls._A_p, nls.q1_center, nls.q2_center,
                                          nls.p1_center, nls.p2_center, nls.p3_center,
                                          E1, E2, E3, B1, B2, B3,
                                          nls.physical_system.params
                                         )


        f_ana = af.broadcast(initialize_f, nls.q1_center, nls.q2_center,
                             addition(nls.p1_center, - A_p1 * t0), 
                             addition(nls.p2_center, - A_p2 * t0),
                             nls.p3_center, nls.physical_system.params
                            )
        
        # f_at_desired_q2 = af.moddims(f_ana[:, 1, 1],
        #                              nls.N_p1, nls.N_p2
        #                             )

        # fig = pl.figure()

        # ax1 = fig.add_subplot(1,2,1)
        # ax1.set_aspect('equal')
        # c1 = ax1.contourf(p1, p2, np.array(f_at_desired_q1), np.linspace(minf, maxf, 120), cmap='bwr')

        # fig.colorbar(c1, orientation = 'vertical', ticks = [minf, 0.5 * (maxf + minf), maxf], fraction=0.046, pad=0.04)

        # ax2 = fig.add_subplot(1,2,2)
        # ax2.set_aspect('equal')
        # c2 = ax2.contourf(p1, p2, np.array(f_at_desired_q2), np.linspace(minf, maxf, 120), cmap='bwr')

        # fig.colorbar(c2, orientation = 'vertical', ticks = [minf, 0.5 * (maxf + minf), maxf], fraction=0.046, pad=0.04)

        # fig.suptitle('Time = %.3f'%(t0))
        # pl.savefig('images/' + '%04d'%time_index + '.png')
        # pl.close(fig)
        # pl.clf()

    nls.f = f_initial
    # In[11]:


    # pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Solver')
    # pl.plot(time_array, rho_data_nls, label='Nonlinear Solver')
    # pl.ylabel(r'MAX($\rho$)')
    # pl.xlabel('Time')
    # pl.legend()


    # In[12]:


    # f_at_desired_q = af.moddims(f[60][:, N_g, N_g + nls.N_q2/2],
    #                             nls.N_p1, nls.N_p2
    #                            )

    # f_initial_at_desired_q = \
    #    initialize.initialize_f(nls.q1_center, nls.q2_center, nls.p1, nls.p2, nls.p3, params)
        
    # f_initial_at_desired_q = af.moddims(f_initial_at_desired_q,
    #                                    nls.N_p1, nls.N_p2
    #                                   )


    # pl.contourf(p1, p2, np.array(f_at_desired_q), 100, cmap='bwr')
    # pl.gca().set_aspect('equal')




    # In[15]:


    # import os
    # os.system('cd images;ffmpeg -f image2 -i %04d.png -vcodec mpeg4 -mbd rd -trellis 2 -cmp 2 -g 300 -pass 1 -r 25 -b 18000000 movie.mp4')


    # In[16]:

    #f_final = af.flat((0.5 * ls.N_q1 * ls.N_q2 * af.ifft2(ls.Y[:, :, :, 0]))[0, 1, :]) 
    # error[i] = af.mean(af.abs(f_final[:, 0, 1] - f_initial[:, 0, 1]))

    #f_analytic =   1.01 * (1 / (2 * np.pi)) \
    #             * af.exp(-(nls.p1_center+t_final)**2 / 2) \
    #             * af.exp(-(nls.p2_center+2*t_final)**2 / 2)

    # Defining the physical system to be solved:
    # system = physical_system(domain,
    #                          boundary_conditions,
    #                          params,
    #                          initialize,
    #                          advection_terms,
    #                          collision_operator.BGK,
    #                          moment_defs
    #                         )

    # N_g_q = system.N_ghost_q

    # nls2 = nonlinear_solver(system)
    # nls2.lie_timestep(t_final)
    error[i] = af.mean(af.abs(nls.f[:, 1, 1] - f_ana[:, 1, 1]))

    # params.solver_method_in_q = 'FVM'
    # params.solver_method_in_p = 'FVM'

print(error)
print(np.polyfit(np.log10(N), np.log10(error), 1))

pl.loglog(N, error, '-o', label = 'Numerical')
pl.loglog(N, error[0] * 32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('plot.png')
