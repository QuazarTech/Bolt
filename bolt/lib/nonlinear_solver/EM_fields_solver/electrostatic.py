#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
from numpy.fft import fftfreq
from bolt.lib.nonlinear_solver.FVM_solver.reconstruct import reconstruct
import params

class poisson_eqn_3D(object):
    """
    This user class is an application context for the problem at hand;
    It contains some parameters and frames the matrix system depending on
    the system state. The Poisson object is used by the
    compute_electrostatic_fields function in computing the electrostatic fields
    using the PETSc's SNES solver methods
    """
    def __init__(self, nonlinear_solver_obj):
        self.da_3D     = nonlinear_solver_obj._da_snes
        self.da_2D     = nonlinear_solver_obj._da_f
        self.obj       = nonlinear_solver_obj
        self.glob_phi  = self.da_3D.createGlobalVec()
        self.local_phi = self.da_3D.createLocalVec() # phi with ghost zones
        
        self.glob_residual  = self.da_3D.createGlobalVec()

        self.N_ghost   = self.obj.N_ghost
        self.N_ghost_poisson = self.obj.N_ghost_poisson
        self.dq1       = self.obj.dq1
        self.dq2       = self.obj.dq2
        self.dq3       = self.obj.dq3

        ((i_q1_2D_start, i_q2_2D_start), 
         (N_q1_2D_local, N_q2_2D_local)
        ) = self.da_2D.getCorners()

        ((i_q1_3D_start, i_q2_3D_start, i_q3_3D_start),
         (N_q1_3D_local, N_q2_3D_local, N_q3_3D_local)
        ) = self.da_3D.getCorners()
        
        self.i_q1_3D_start = i_q1_3D_start
        self.i_q2_3D_start = i_q2_3D_start
        self.i_q3_3D_start = i_q3_3D_start

        self.i_q1_3D_end = i_q1_3D_start + N_q1_3D_local - 1
        self.i_q2_3D_end = i_q2_3D_start + N_q2_3D_local - 1
        self.i_q3_3D_end = i_q3_3D_start + N_q3_3D_local - 1

        self.N_q1_2D_local = N_q1_2D_local
        self.N_q2_2D_local = N_q2_2D_local

        self.N_q1_3D_local = N_q1_3D_local
        self.N_q2_3D_local = N_q2_3D_local
        self.N_q3_3D_local = N_q3_3D_local

        location_in_q3 = self.obj.location_in_q3
        N_g  = self.N_ghost         # Ghost zones of Boltzmann solver
        N_gp = self.N_ghost_poisson # Ghost zones of Poisson solver 

        self.density_np = np.zeros([N_q2_2D_local + 2*N_g,
                                    N_q1_2D_local + 2*N_g 
				   ]
				  )
        # Cell centers in 3D
        i_q1_3D = ( (i_q1_3D_start + 0.5)
                   + np.arange(-N_gp, N_q1_3D_local + N_gp)
                  )

        i_q2_3D = ( (i_q2_3D_start + 0.5)
                   + np.arange(-N_gp, N_q2_3D_local + N_gp)
                  )

        i_q3_3D = ( (i_q3_3D_start + 0.5)
                   + np.arange(-N_gp, N_q3_3D_local + N_gp)
                  )

        q1_2D_start = self.obj.q1_start
        q1_2D_end   = self.obj.q1_end
        q2_2D_start = self.obj.q2_start
        q2_2D_end   = self.obj.q2_end
        q3_3D_start = self.obj.q3_3D_start

        # TODO: Code below is duplication of _calculate_q_center in nonlinear_solver.py
        i_q1_2D = ( (i_q1_2D_start + 0.5)
                   + np.arange(-N_g, N_q1_2D_local + N_g)
                  )

        i_q2_2D = ( (i_q2_2D_start + 0.5)
                   + np.arange(-N_g, N_q2_2D_local + N_g)
                  )

        q1_2D =  q1_2D_start + i_q1_2D * self.dq1
        q2_2D =  q2_2D_start + i_q2_2D * self.dq2

        length_q1_2d = (q1_2D_end - q1_2D_start)
        length_q2_2d = (q2_2D_end - q2_2D_start)

        length_multiples_q1 = self.obj.length_multiples_q1
        length_multiples_q2 = self.obj.length_multiples_q2

        q1_3D =  q1_2D_start - length_multiples_q1*length_q1_2d + i_q1_3D * self.dq1
        q2_3D =  q2_2D_start - length_multiples_q2*length_q2_2d + i_q2_3D * self.dq2
        q3_3D =  q3_3D_start + i_q3_3D * self.dq3

        self.q1_3D = q1_3D
        self.q2_3D = q2_3D
        self.q3_3D = q3_3D

        glob_epsilon  = self.da_3D.createGlobalVec()
        local_epsilon = self.da_3D.createLocalVec()

        epsilon_array = local_epsilon.getArray(readonly=0)
        epsilon_array = epsilon_array.reshape([N_q3_3D_local + 2*N_gp, \
                                               N_q2_3D_local + 2*N_gp, \
                                               N_q1_3D_local + 2*N_gp
                                              ]
                                             )
        epsilon_array[:] = params.epsilon0
        epsilon_array[q3_3D<location_in_q3+self.dq3, :, :] = \
                params.epsilon_relative*params.epsilon0
        self.epsilon_array = epsilon_array

        q2_3D_data_structure, q3_3D_data_structure, q1_3D_data_structure = \
                np.meshgrid(q2_3D, q3_3D, q1_3D)
        
        q1_2D_data_structure, q2_2D_data_structure = \
                np.meshgrid(q1_2D, q2_2D)

        tol = 1e-10
        self.cond_3D =   (q1_3D_data_structure >= q1_2D[N_g] - tol) \
                       & (q1_3D_data_structure <= q1_2D[-1-N_g] + tol) \
                       & (q2_3D_data_structure >= q2_2D[N_g] - tol) \
                       & (q2_3D_data_structure <= q2_2D[-1-N_g] + tol) \
                       & (q3_3D_data_structure >= location_in_q3) \
                       & (q3_3D_data_structure < location_in_q3 + self.dq3)

        self.cond_2D =   (q1_2D_data_structure >= q1_2D[N_g]) \
                       & (q1_2D_data_structure <= q1_2D[-1-N_g]) \
                       & (q2_2D_data_structure >= q2_2D[N_g]) \
                       & (q2_2D_data_structure <= q2_2D[-1-N_g])

        self.cond_3D_phi =   (q1_3D_data_structure >= q1_2D[0] - tol) \
                           & (q1_3D_data_structure <= q1_2D[-1] + tol) \
                           & (q2_3D_data_structure >= q2_2D[0] - tol) \
                           & (q2_3D_data_structure <= q2_2D[-1] + tol) \
                           & (q3_3D_data_structure >= location_in_q3) \
                           & (q3_3D_data_structure < location_in_q3 + self.dq3)

        contact_start = params.contact_start
        contact_end   = params.contact_end

        self.cond_left_contact = \
        (q1_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] < q1_2D[N_g] - tol) \
      & (q2_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] > contact_start-tol) \
      & (q2_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] < contact_end + tol) \
      & (q3_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] > location_in_q3) \
      & (q3_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] < location_in_q3 + self.dq3)

        self.cond_right_contact = \
        (q1_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] > q1_2D[-1-N_g] + tol) \
      & (q2_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] > contact_start-tol) \
      & (q2_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] < contact_end + tol) \
      & (q3_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] > location_in_q3) \
      & (q3_3D_data_structure[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp] < location_in_q3 + self.dq3)

        backgate_potential = params.backgate_potential

        self.q3    = q3_3D_data_structure
        z          = self.q3
        #z_sample   = q3_3D[self.q3_2D_in_3D_index_start]
        #TODO: FIX THIS ASAP
        z_sample   = location_in_q3
        z_backgate = q3_3D[N_gp]
        side_wall_boundaries = \
	    backgate_potential*(z_sample - z)/(z_sample - z_backgate)

        self.bc = 0.*self.q3 # 3D boundary condition array

        self.bc[:]          = 0.
        self.bc[:N_gp, :, :] = backgate_potential # backgate

        below_sample = q3_3D<location_in_q3+self.dq3

        # Back wall
        self.bc[below_sample, :N_gp, :]               = \
            side_wall_boundaries[below_sample, :N_gp, :]

        # Front wall
        self.bc[below_sample, N_q2_3D_local+N_gp:, :] = \
            side_wall_boundaries[below_sample, N_q2_3D_local+N_gp:, :]

        # Left wall
        self.bc[below_sample, :, :N_gp]               = \
            side_wall_boundaries[below_sample, :, :N_gp]

        # Right wall
        self.bc[below_sample, :, N_q1_3D_local+N_gp:] = \
            side_wall_boundaries[below_sample, :, N_q1_3D_local+N_gp:]

        return


    def compute_residual(self, snes, phi, residual):
        self.da_3D.globalToLocal(phi, self.local_phi)

        N_g  = self.N_ghost
        N_gp = self.N_ghost_poisson 

        N_q1_2D_local = self.N_q1_2D_local
        N_q2_2D_local = self.N_q2_2D_local

        N_q1_3D_local = self.N_q1_3D_local
        N_q2_3D_local = self.N_q2_3D_local
        N_q3_3D_local = self.N_q3_3D_local

        phi_array = self.local_phi.getArray(readonly=0)
        phi_array = phi_array.reshape([N_q3_3D_local + 2*N_gp, \
                                       N_q2_3D_local + 2*N_gp, \
                                       N_q1_3D_local + 2*N_gp
        			      ]
                                     )

        phi_glob_array = phi.getArray(readonly=0)
        phi_glob_array = phi_glob_array.reshape([N_q3_3D_local, \
                                                 N_q2_3D_local, \
                                                 N_q1_3D_local
        		                 	]
                                               )
    
        residual_array = residual.getArray(readonly=0)
        residual_array = residual_array.reshape([N_q3_3D_local, \
                                                 N_q2_3D_local, \
                                                 N_q1_3D_local
        					]
                                               )
        # Boundary conditions
        bc = self.bc
        if (self.i_q3_3D_start==0):
            phi_array[:N_gp, :, :]               = bc[:N_gp, :, :]

        if (self.i_q3_3D_end==self.obj.N_q3_poisson-1):
            phi_array[N_q3_3D_local+N_gp:, :, :] = bc[N_q3_3D_local+N_gp:, :, :]

        if (self.i_q2_3D_start==0):
            phi_array[:, :N_gp, :]               = bc[:, :N_gp, :]

        if (self.i_q2_3D_end==self.obj.N_q2_poisson-1):
            phi_array[:, N_q2_3D_local+N_gp:, :] = bc[:, N_q2_3D_local+N_gp:, :]

        if (self.i_q1_3D_start==0):
            phi_array[:, :, :N_gp]               = bc[:, :, :N_gp]

        if (self.i_q1_3D_end==self.obj.N_q1_poisson-1):
            phi_array[:, :, N_q1_3D_local+N_gp:] = bc[:, :, N_q1_3D_local+N_gp:]

        phi_plus_x  = np.roll(phi_array, shift=-1, axis=2) # (i+3/2, j+1/2, k+1/2)
        phi_minus_x = np.roll(phi_array, shift=1,  axis=2) # (i-1/2, j+1/2, k+1/2)
        phi_plus_y  = np.roll(phi_array, shift=-1, axis=1) # (i+1/2, j+3/2, k+1/2)
        phi_minus_y = np.roll(phi_array, shift=1,  axis=1) # (i+1/2, j-1/2, k+1/2)
        phi_plus_z  = np.roll(phi_array, shift=-1, axis=0) # (i+1/2, j+1/2, k+3/2)
        phi_minus_z = np.roll(phi_array, shift=1,  axis=0) # (i+1/2, j+1/2, k+3/2)

        epsilon_array  = self.epsilon_array
        eps_left_edge  = epsilon_array # (i, j+1/2, k+1/2)
        eps_right_edge = np.roll(epsilon_array, shift=-1, axis=2) # (i+1, j+1/2, k+1/2)

        eps_bot_edge   = epsilon_array # (i+1/2, j, k+1/2)
        eps_top_edge   = np.roll(epsilon_array, shift=-1, axis=1) # (i+1/2, j+1, k+1/2)

        eps_back_edge  = epsilon_array # (i+1/2, j+1/2, k)
        eps_front_edge = np.roll(epsilon_array, shift=-1, axis=0) # (i+1/2, j+1/2, k+1)

        D_left_edge  = eps_left_edge  * (phi_array  - phi_minus_x)/self.dq1
        D_right_edge = eps_right_edge * (phi_plus_x - phi_array)  /self.dq1

        D_bot_edge   = eps_bot_edge   * (phi_array  - phi_minus_y)/self.dq2
        D_top_edge   = eps_top_edge   * (phi_plus_y - phi_array  )/self.dq2

        D_back_edge  = eps_back_edge  * (phi_array  - phi_minus_z)/self.dq3
        D_front_edge = eps_front_edge * (phi_plus_z - phi_array  )/self.dq3

        laplacian_phi =  (D_right_edge - D_left_edge) /self.dq1 \
                       + (D_top_edge   - D_bot_edge)  /self.dq2 \
                       + (D_front_edge - D_back_edge) /self.dq3

        if (params.solve_for_equilibrium):
            # solve \nabla^2 phi = -rho
            # where rho = \int f_FD(mu) d^3k with mu - e * phi = const

            mu =  params.global_chem_potential \
                + params.charge_electron*phi_array[self.cond_3D]
            mu = af.to_array(mu)
            mu = af.moddims(af.to_array(mu),
                            1, 
                            self.obj.N_q1_local, 
                            self.obj.N_q2_local
                           )
            self.obj.f[:, N_g:-N_g, N_g:-N_g] = \
                params.fermi_dirac(mu, params.E_band)

            density_af = af.moddims(self.obj.compute_moments('density'),
                                      (self.obj.N_q1_local+2*self.obj.N_ghost)
                                    * (self.obj.N_q2_local+2*self.obj.N_ghost)
                                   )
            density_np = density_af.to_ndarray()
            density_np =\
                density_np.reshape([self.obj.N_q2_local+2*self.obj.N_ghost,
                                    self.obj.N_q1_local+2*self.obj.N_ghost
                                   ]
                                  )

            laplacian_phi[self.cond_3D] += \
                  -params.charge_electron \
                 * density_np[self.cond_2D] / self.dq3
        else:
            # Density is an external source. Not explicitly coupled to phi.
            laplacian_phi[self.cond_3D] += \
                  -params.charge_electron \
                 * self.density_np[self.cond_2D] / self.dq3

        residual_array[:, :, :] = \
            laplacian_phi[N_gp:-N_gp, N_gp:-N_gp, N_gp:-N_gp]

        residual_array[self.cond_left_contact] = \
                phi_glob_array[self.cond_left_contact] - 0.

        residual_array[self.cond_right_contact] = \
                phi_glob_array[self.cond_right_contact] - 0.
        return

def compute_electrostatic_fields(self):
    
    if(self.performance_test_flag == True):
        tic = af.time()

    N_g  = self.N_ghost
    N_gp = self.poisson.N_ghost_poisson

    density_af = af.moddims(self.compute_moments('density'),
                              (self.N_q1_local+2*N_g)
                            * (self.N_q2_local+2*N_g)
                           )
    density_np = density_af.to_ndarray()
    self.poisson.density_np[:] =\
        density_np.reshape([self.N_q2_local+2*N_g,
                            self.N_q1_local+2*N_g
                           ]
                          )

    self.snes.solve(None, self.poisson.glob_phi)

    self._da_snes.globalToLocal(self.poisson.glob_phi, 
                                self.poisson.local_phi
			       )
    phi_local_array = self.poisson.local_phi.getArray()
    phi_local_array = \
        phi_local_array.reshape([self.poisson.N_q3_3D_local+2*N_gp,
                                 self.poisson.N_q2_3D_local+2*N_gp,
                                 self.poisson.N_q1_3D_local+2*N_gp
                                ]
                               )

    phi_2D_local_array = phi_local_array[self.poisson.cond_3D_phi]

    # convert from np->af
    self.phi = af.to_array(phi_2D_local_array) 
    # Since rho was defined at (i + 0.5, j + 0.5)
    # Electric Potential returned will also be at (i + 0.5, j + 0.5)
    self.phi = af.moddims(self.phi,
                          self.N_q1_local + 2*N_g,
                          self.N_q2_local + 2*N_g
                         )
    params.phi = self.phi

    # Obtaining the electric field values at (i+0.5, j+0.5) by first
    # reconstructing phi to the edges and then differencing them. Needed
    # in case phi develops large gradients

    method_in_q = self.physical_system.params.reconstruction_method_in_q

    phi_left, phi_right = reconstruct(self, self.phi, 0, method_in_q)
    E1 = -(phi_right - phi_left)/self.dq1

    phi_bottom, phi_top = reconstruct(self, self.phi, 1, method_in_q)
    E2 = -(phi_top - phi_bottom)/self.dq2

    af.eval(E1, E2)

    E1 = af.moddims(E1,
                    1,
                    self.N_q1_local + 2*N_g,
                    self.N_q2_local + 2*N_g
                   )

    E2 = af.moddims(E2,
                    1,
                    self.N_q1_local + 2*N_g,
                    self.N_q2_local + 2*N_g
                   )

    self.cell_centered_EM_fields[0, :] = E1
    self.cell_centered_EM_fields[1, :] = E2
    self.cell_centered_EM_fields[2, :] = 0. # TODO: worry about this later

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic

    return


def fft_poisson(self, f=None):
    """
    Solves the Poisson Equation using the FFTs:

    Used as a backup solver in case of low resolution runs
    (ie. used on a single node) with periodic boundary
    conditions.
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    if (self._comm.size != 1):
        raise Exception('FFT solver can only be used when run in serial')

    else:
        N_g = self.N_ghost
        rho = af.reorder(  self.physical_system.params.charge_electron \
                         * self.compute_moments('density', f)[:, N_g:-N_g, N_g:-N_g],
                         1, 2, 0
                        )

        k_q1 = fftfreq(rho.shape[0], self.dq1)
        k_q2 = fftfreq(rho.shape[1], self.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q1 = af.to_array(k_q1)
        k_q2 = af.to_array(k_q2)

        rho_hat = af.fft2(rho)

        potential_hat       = rho_hat / (4 * np.pi**2 * (k_q1**2 + k_q2**2))
        potential_hat[0, 0] = 0

        E1_hat = -1j * 2 * np.pi * k_q1 * potential_hat
        E2_hat = -1j * 2 * np.pi * k_q2 * potential_hat

        # Non-inclusive of ghost-zones:
        E1_physical = af.reorder(af.real(af.ifft2(E1_hat)), 2, 0, 1)
        E2_physical = af.reorder(af.real(af.ifft2(E2_hat)), 2, 0, 1)

        self.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g] = E1_physical
        self.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g] = E2_physical

        af.eval(self.cell_centered_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
