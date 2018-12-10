#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This checks that the explicit time-stepping of the
FDTD algorithm works as intended. Since Maxwell's
equation have wave like solutions, in this test we evolve
the initial state for a single timeperiod and compare the
final solution state with the initial state.

We check the fall off in error with the increase in resolution
(convergence rate) to validate the explicit FDTD algorithm.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc
# import matplotlib as mpl
# mpl.use('agg')
import pylab as pl

from bolt.lib.nonlinear.fields.fields import fields_solver
from bolt.lib.physical_system import physical_system

from input_files import domain
from input_files import params
from input_files import initialize_fdtd_mode1
from input_files import initialize_fdtd_mode2
from input_files import boundary_conditions
from input_files import boundary_conditions_mirror

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'gist_heat'
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

class test_periodic(object):

    def __init__(self, N, initialize, params):

        domain.N_q1 = int(N)
        domain.N_q2 = int(N)

        N_g    = domain.N_ghost
        system = physical_system(domain,
                                 boundary_conditions,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        self.fields_solver = fields_solver(system, 
                                           af.constant(0, 1, 1, 
                                                       domain.N_q1 + 2 * N_g, 
                                                       domain.N_q2 + 2 * N_g, 
                                                       dtype = af.Dtype.f64
                                                      )
                                          )

        return

class test_mirror(object):

    def __init__(self, N, initialize, params):

        domain.N_q1 = int(N)
        domain.N_q2 = int(N)

        N_g    = domain.N_ghost
        system = physical_system(domain,
                                 boundary_conditions_mirror,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        self.fields_solver = fields_solver(system, 
                                           af.constant(0, 1, 1, 
                                                       domain.N_q1 + 2 * N_g, 
                                                       domain.N_q2 + 2 * N_g, 
                                                       dtype = af.Dtype.f64
                                                      )
                                          )

        return

def test_fdtd_mode1_periodic():

    N = np.array([64]) #2**np.arange(5, 8)

    error_B1 = np.zeros(N.size)
    error_B2 = np.zeros(N.size)
    error_E3 = np.zeros(N.size)

    for i in range(N.size):

        af.device_gc()
        dt   = (1 / int(N[i])) * np.sqrt(9/5) / 2
        time = np.arange(dt, 10 * np.sqrt(9/5) + dt, dt)

        params.dt = dt

        obj = test_periodic(int(N[i]), initialize_fdtd_mode1, params)
        N_g = obj.fields_solver.N_g

        E1_initial = obj.fields_solver.yee_grid_EM_fields[0].copy()
        E2_initial = obj.fields_solver.yee_grid_EM_fields[1].copy()
        E3_initial = obj.fields_solver.yee_grid_EM_fields[2].copy()

        B1_initial = obj.fields_solver.yee_grid_EM_fields[3].copy()
        B2_initial = obj.fields_solver.yee_grid_EM_fields[4].copy()
        B3_initial = obj.fields_solver.yee_grid_EM_fields[5].copy()

        # electric_energy = 1/4 * (  E3_initial**2                       # Ez(i+1/2, j+1/2)
        #                          + af.shift(E3_initial, 0, 0, 1, 0)**2 # Ez(i-1/2, j+1/2)
        #                          + af.shift(E3_initial, 0, 0, 0, 1)**2 # Ez(i+1/2, j-1/2)
        #                          + af.shift(E3_initial, 0, 0, 1, 1)**2 # Ez(i-1/2, j-1/2)
        #                         )

        # magnetic_energy_x = 0.5 * (  B1_n_plus_half * B1_n_minus_half                        # (i+1/2, j)
        #                            + af.shift(B1_n_plus_half * B1_n_minus_half, 0, 0, 1, 0)  # (i-1/2, j)
        #                           )

        # magnetic_energy_y = 0.5 * (  B2_n_plus_half * B2_n_minus_half                       # (i, j+1/2)
        #                            + af.shift(B2_n_plus_half * B2_n_minus_half, 0, 0, 0, 1) # (i, j-1/2)
        #                           )

        error    = np.zeros([time.size])
        for time_index, t0 in enumerate(time):

            B1_n_minus_half = obj.fields_solver.yee_grid_EM_fields[3].copy()
            B2_n_minus_half = obj.fields_solver.yee_grid_EM_fields[4].copy()

            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

            B1_n_plus_half = obj.fields_solver.yee_grid_EM_fields[3].copy()
            B2_n_plus_half = obj.fields_solver.yee_grid_EM_fields[4].copy()

            E3_at_n = obj.fields_solver.yee_grid_EM_fields[2].copy()

            electric_energy   = E3_at_n**2
            magnetic_energy_x = B1_n_plus_half * B1_n_minus_half
            magnetic_energy_y = B2_n_plus_half * B2_n_minus_half

            # pl.plot(np.array(obj.fields_solver.q1_center).reshape(134, 9)[3:-3, 0],
            #         np.array(obj.fields_solver.yee_grid_EM_fields[1]).reshape(134, 9)[3:-3, 0],
            #         label = r'$Ey_{i+1/2}^n$')
            # pl.plot(np.array(obj.fields_solver.q1_center).reshape(134, 9)[3:-3, 0],
            #         np.array(obj.fields_solver.yee_grid_EM_fields[5]).reshape(134, 9)[3:-3, 0], '--',
            #         label = r'${Bz}_{i}^{n+1/2}$')
            # pl.legend(fontsize = 20)
            # pl.ylim([-2, 2])
            # pl.title('Time = %.2f'%t0)
            # pl.savefig('images/%04d'%time_index + '.png')
            # pl.clf()

            # pl.contourf(np.array(obj.fields_solver.yee_grid_EM_fields[2]).reshape(134, 134), 40)
            # pl.savefig('images/%04d'%time_index + '.png')
            # pl.clf()

            energy = af.sum((electric_energy + magnetic_energy_x + magnetic_energy_y)[:, :, N_g:-N_g, N_g:-N_g])
            if(time_index == 0):
                error[time_index] = energy * obj.fields_solver.dq1 * obj.fields_solver.dq2
            else:
                error[time_index] = abs((energy) * obj.fields_solver.dq1 * obj.fields_solver.dq2 -  error[0])

        pl.semilogy(time[1:], error[1:])
        pl.ylabel('Error')
        pl.xlabel('Time')
        pl.ylim([1e-15, 1e-9])
        pl.savefig('plot.png', bbox_inches = 'tight')


        error_B1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[3, :, N_g:-N_g, N_g:-N_g] -
                                    B1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B1_initial.elements())

        error_B2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[4, :, N_g:-N_g, N_g:-N_g] -
                                    B2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B2_initial.elements())

        error_E3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[2, :, N_g:-N_g, N_g:-N_g] -
                                    E3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E3_initial.elements())

    poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
    poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
    poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

    print(error_B1)
    print(error_B2)
    print(error_E3)

    print(poly_B1)
    print(poly_B2)
    print(poly_E3)

    pl.loglog(N, error_B1, '-o', label = r'$B_x$')
    pl.loglog(N, error_B2, '-o', label = r'$B_y$')
    pl.loglog(N, error_E3, '-o', label = r'$E_z$')
    pl.loglog(N, error_B2[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig('convergenceplot.png')

    assert (abs(poly_B1[0] + 2) < 0.2)
    assert (abs(poly_B2[0] + 2) < 0.2) 
    assert (abs(poly_E3[0] + 2) < 0.2)

def test_fdtd_mode2_periodic():

    N = np.array([64]) #2**np.arange(5, 8)

    error_E1 = np.zeros(N.size)
    error_E2 = np.zeros(N.size)
    error_B3 = np.zeros(N.size)

    for i in range(N.size):

        dt   = (1 / int(N[i])) * np.sqrt(9/5) / 2
        time = np.arange(dt, 10 * np.sqrt(9/5) + dt, dt)

        params.dt = dt

        obj = test_periodic(N[i], initialize_fdtd_mode2, params)
        N_g = obj.fields_solver.N_g

        B3_initial = obj.fields_solver.yee_grid_EM_fields[5].copy()
        E1_initial = obj.fields_solver.yee_grid_EM_fields[0].copy()
        E2_initial = obj.fields_solver.yee_grid_EM_fields[1].copy()

        error = np.zeros([time.size])
        for time_index, t0 in enumerate(time):

            B3_n_minus_half = obj.fields_solver.yee_grid_EM_fields[5].copy()
            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)
            B3_n_plus_half = obj.fields_solver.yee_grid_EM_fields[5].copy()

            E1_at_n = obj.fields_solver.yee_grid_EM_fields[0].copy()
            E2_at_n = obj.fields_solver.yee_grid_EM_fields[1].copy()

            electric_energy_x = E1_at_n**2
            electric_energy_y = E2_at_n**2
            magnetic_energy   = B3_n_plus_half * B3_n_minus_half

            energy = af.sum((magnetic_energy + electric_energy_x + electric_energy_y)[:, :, N_g:-N_g, N_g:-N_g])
            if(time_index == 0):
                error[time_index] = energy * obj.fields_solver.dq1 * obj.fields_solver.dq2
            else:
                error[time_index] = abs((energy) * obj.fields_solver.dq1 * obj.fields_solver.dq2 -  error[0])

        pl.semilogy(time[1:], error[1:])
        pl.ylabel('Error')
        pl.xlabel('Time')
        pl.ylim([1e-15, 1e-9])
        pl.savefig('plot.png', bbox_inches = 'tight')

        error_E1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[0, :, N_g:-N_g, N_g:-N_g] -
                                    E1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E1_initial.elements())

        error_E2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[1, :, N_g:-N_g, N_g:-N_g] -
                                    E2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E2_initial.elements())

        error_B3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[5, :, N_g:-N_g, N_g:-N_g] -
                                    B3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B3_initial.elements())

    print(error_E1)
    print(error_E2)
    print(error_B3)

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
    poly_B3 = np.polyfit(np.log10(N), np.log10(error_B3), 1)

    pl.loglog(N, error_E1, '-o', label = r'$E_x$')
    pl.loglog(N, error_E2, '-o', label = r'$E_y$')
    pl.loglog(N, error_B3, '-o', label = r'$B_z$')
    pl.loglog(N, error_E1[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig('convergenceplot.png')

    print(poly_E1)
    print(poly_E2)
    print(poly_B3)

    assert (abs(poly_E1[0] + 2) < 0.3)
    assert (abs(poly_E2[0] + 2) < 0.3)
    assert (abs(poly_B3[0] + 2) < 0.3)

def test_fdtd_mode1_mirror():

    N = np.array([128]) #2**np.arange(5, 11)

    error_B1 = np.zeros(N.size)
    error_B2 = np.zeros(N.size)
    error_E3 = np.zeros(N.size)

    for i in range(N.size):

        dt   = (1 / int(N[i])) *  np.sqrt(9 / 5) / 2
        time = np.arange(dt, 4 * np.sqrt(9 / 5) + dt, dt)

        params.dt = dt

        obj = test_mirror(N[i], initialize_fdtd_mode1, params)
        N_g = obj.fields_solver.N_g

        E3_initial = obj.fields_solver.yee_grid_EM_fields[2].copy()
        B1_initial = obj.fields_solver.yee_grid_EM_fields[3].copy()
        B2_initial = obj.fields_solver.yee_grid_EM_fields[4].copy()

        for time_index, t0 in enumerate(time):
            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)
        
            # pl.plot(np.array(obj.fields_solver.yee_grid_EM_fields[2]).reshape(134, 9)[3:-3, 0], label = r'$E_z$')
            # pl.plot(np.array(obj.fields_solver.yee_grid_EM_fields[4]).reshape(134, 9)[3:-3, 0], label = r'$B_y$')
            # pl.legend()
            # pl.ylim([-2, 2])
            # pl.title('Time = %.2f'%t0)
            # pl.savefig('images/%04d'%time_index + '.png')
            # pl.clf()
            
        error_B1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[3, :, N_g:-N_g, N_g:-N_g] -
                                    B1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B1_initial.elements())

        error_B2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[4, :, N_g:-N_g, N_g:-N_g] -
                                    B2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B2_initial.elements())

        error_E3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[2, :, N_g:-N_g, N_g:-N_g] -
                                    E3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E3_initial.elements())

    poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
    poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
    poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

    print(error_B1)
    print(error_B2)
    print(error_E3)

    pl.loglog(N, error_B1, '-o', label = r'$B_x$')
    pl.loglog(N, error_B2, '-o', label = r'$B_y$')
    pl.loglog(N, error_E3, '-o', label = r'$E_z$')
    pl.loglog(N, error_B1[0]*N[0]**1/N**1, '--', color = 'black', 
              label = r'$\mathcal{O}(N^{-1})$'
             )
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend(framealpha = 0)
    pl.savefig('convergenceplot.png', bbox_inches = 'tight')

    print(poly_B1)
    print(poly_B2)
    print(poly_E3)

    assert (abs(poly_B1[0] + 1) < 0.2)
    assert (abs(poly_B2[0] + 1) < 0.2) 
    assert (abs(poly_E3[0] + 1) < 0.2)

def test_fdtd_mode2_mirror():

    N = 2**np.arange(5, 8)

    error_E1 = np.zeros(N.size)
    error_E2 = np.zeros(N.size)
    error_B3 = np.zeros(N.size)

    for i in range(N.size):

        dt   = (1 / int(N[i])) * np.sqrt(9/5) / 2
        time = np.arange(dt, 4 * np.sqrt(9/5) + dt, dt)

        params.dt = dt

        obj = test_mirror(N[i], initialize_fdtd_mode2, params)
        N_g = obj.fields_solver.N_g

        B3_initial = obj.fields_solver.yee_grid_EM_fields[5].copy()
        E1_initial = obj.fields_solver.yee_grid_EM_fields[0].copy()
        E2_initial = obj.fields_solver.yee_grid_EM_fields[1].copy()

        for time_index, t0 in enumerate(time):
            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

        error_E1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[0, :, N_g:-N_g, N_g:-N_g] -
                                    E1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E1_initial.elements())

        error_E2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[1, :, N_g:-N_g, N_g:-N_g] -
                                    E2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E2_initial.elements())

        error_B3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[5, :, N_g:-N_g, N_g:-N_g] -
                                    B3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B3_initial.elements())

    print(error_E1)
    print(error_E2)
    print(error_B3)

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
    poly_B3 = np.polyfit(np.log10(N), np.log10(error_B3), 1)

    pl.loglog(N, error_E1, '-o', label = r'$E_x$')
    pl.loglog(N, error_E2, '-o', label = r'$E_y$')
    pl.loglog(N, error_B3, '-o', label = r'$B_z$')
    pl.loglog(N, error_E1[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig('convergenceplot.png')

    assert (abs(poly_E1[0] + 1) < 0.2)
    assert (abs(poly_E2[0] + 1) < 0.2)
    assert (abs(poly_B3[0] + 1) < 0.2)


test_fdtd_mode1_periodic()
