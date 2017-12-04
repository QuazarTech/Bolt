#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import petsc4py, sys 
petsc4py.init(sys.argv)
from petsc4py import PETSc
import arrayfire as af
import numpy as np
from numpy.fft import fftfreq
import pylab as pl
import params

pl.rcParams['figure.figsize']  = 17, 7.5
pl.rcParams['figure.dpi']      = 150
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
        self.dq1       = self.obj.dq1
        self.dq2       = self.obj.dq2
        self.dq3       = self.obj.dq3

        ((i_q1_2D_start, i_q2_2D_start), 
         (N_q1_2D_local, N_q2_2D_local)
        ) = self.da_2D.getCorners()

        ((i_q1_3D_start, i_q2_3D_start, i_q3_3D_start),
         (N_q1_3D_local, N_q2_3D_local, N_q3_3D_local)
        ) = self.da_3D.getCorners()

        self.N_q1_2D_local = N_q1_2D_local
        self.N_q2_2D_local = N_q2_2D_local

        self.N_q1_3D_local = N_q1_3D_local
        self.N_q2_3D_local = N_q2_3D_local
        self.N_q3_3D_local = N_q3_3D_local

        location_in_q3 = 10.
        N_g = self.N_ghost

        self.density_np = np.zeros([N_q2_2D_local + 2*N_g,
                                    N_q1_2D_local + 2*N_g 
				   ]
				  )
        # Cell centers in 3D
        i_q1_3D = ( (i_q1_3D_start + 0.5)
                   + np.arange(-N_g, N_q1_3D_local + N_g)
                  )

        i_q2_3D = ( (i_q2_3D_start + 0.5)
                   + np.arange(-N_g, N_q2_3D_local + N_g)
                  )

        i_q3_3D = ( (i_q3_3D_start + 0.5)
                   + np.arange(-N_g, N_q3_3D_local + N_g)
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

        self.q3_2D_in_3D_index_start = np.where(q3_3D > location_in_q3 - self.dq3)[0][0]
        self.q3_2D_in_3D_index_end   = np.where(q3_3D < location_in_q3 + self.dq3)[0][-1]

        self.q1_2D_in_3D_index_start = np.where(abs(q1_3D - q1_2D[N_g] )   < 1e-10)[0][0]
        self.q1_2D_in_3D_index_end   = np.where(abs(q1_3D - q1_2D[-1-N_g]) < 1e-10)[0][0]
        self.q2_2D_in_3D_index_start = np.where(abs(q2_3D - q2_2D[N_g] )   < 1e-10)[0][0]
        self.q2_2D_in_3D_index_end   = np.where(abs(q2_3D - q2_2D[-1-N_g]) < 1e-10)[0][0]

        glob_epsilon  = self.da_3D.createGlobalVec()
        local_epsilon = self.da_3D.createLocalVec()

        epsilon_array = local_epsilon.getArray(readonly=0)
        epsilon_array = epsilon_array.reshape([N_q3_3D_local + 2*N_g, \
                                               N_q2_3D_local + 2*N_g, \
                                               N_q1_3D_local + 2*N_g
                                              ]
                                             )
        epsilon_array[:] = params.epsilon0
        epsilon_array[:self.q3_2D_in_3D_index_start, :, :] = 10.*params.epsilon0
        self.epsilon_array = epsilon_array
        
        q3_3D_data_structure = 0.*epsilon_array

        for j in range(q3_3D_data_structure.shape[1]):
            for i in range(q3_3D_data_structure.shape[2]):
                q3_3D_data_structure[:, j, i] = q3_3D
        
        backgate_potential = -100.

        self.q3    = q3_3D_data_structure
        z          = self.q3
        z_sample   = q3_3D[self.q3_2D_in_3D_index_start]
        z_backgate = q3_3D[0]
        side_wall_boundaries = \
	    backgate_potential*(z_sample - z)/(z_sample - z_backgate)

        self.bc = 0.*self.q3 # 3D boundary condition array

        self.bc[:]          = 0.
        self.bc[:N_g, :, :] = backgate_potential # backgate

        self.bc[:self.q3_2D_in_3D_index_start, :N_g, :]               = \
            side_wall_boundaries[:self.q3_2D_in_3D_index_start, :N_g, :]

        self.bc[:self.q3_2D_in_3D_index_start, N_q2_3D_local+N_g:, :] = \
            side_wall_boundaries[:self.q3_2D_in_3D_index_start, N_q2_3D_local+N_g:, :]

        self.bc[:self.q3_2D_in_3D_index_start, :, :N_g]               = \
            side_wall_boundaries[:self.q3_2D_in_3D_index_start, :, :N_g]

        self.bc[:self.q3_2D_in_3D_index_start, :, N_q1_3D_local+N_g:] = \
            side_wall_boundaries[:self.q3_2D_in_3D_index_start, :, N_q1_3D_local+N_g:]
        
        return


    def compute_residual(self, snes, phi, residual):
        self.da_3D.globalToLocal(phi, self.local_phi)

        N_g = self.N_ghost

        N_q1_2D_local = self.N_q1_2D_local
        N_q2_2D_local = self.N_q2_2D_local

        N_q1_3D_local = self.N_q1_3D_local
        N_q2_3D_local = self.N_q2_3D_local
        N_q3_3D_local = self.N_q3_3D_local

        phi_array = self.local_phi.getArray(readonly=0)
        phi_array = phi_array.reshape([N_q3_3D_local + 2*N_g, \
                                       N_q2_3D_local + 2*N_g, \
                                       N_q1_3D_local + 2*N_g
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
        phi_array[:N_g, :, :]               = bc[:N_g, :, :]
        phi_array[N_q3_3D_local+N_g:, :, :] = bc[N_q3_3D_local+N_g:, :, :]
        phi_array[:, :N_g, :]               = bc[:, :N_g, :]
        phi_array[:, N_q2_3D_local+N_g:, :] = bc[:, N_q2_3D_local+N_g:, :]
        phi_array[:, :, :N_g]               = bc[:, :, :N_g]
        phi_array[:, :, N_q1_3D_local+N_g:] = bc[:, :, N_q1_3D_local+N_g:]


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

        q3_2D_in_3D_index_start = self.q3_2D_in_3D_index_start 
        q3_2D_in_3D_index_end   = self.q3_2D_in_3D_index_end   
                                                               
        q1_2D_in_3D_index_start = self.q1_2D_in_3D_index_start 
        q1_2D_in_3D_index_end   = self.q1_2D_in_3D_index_end   
        q2_2D_in_3D_index_start = self.q2_2D_in_3D_index_start 
        q2_2D_in_3D_index_end   = self.q2_2D_in_3D_index_end   

        laplacian_phi[q3_2D_in_3D_index_start,
                      q2_2D_in_3D_index_start:q2_2D_in_3D_index_end+1,
                      q1_2D_in_3D_index_start:q1_2D_in_3D_index_end+1
                     ] \
                     += \
              -params.charge_electron \
             * self.density_np[N_g:-N_g, N_g:-N_g] / self.dq3

        residual_array[:, :, :] = \
            laplacian_phi[N_g:-N_g, N_g:-N_g, N_g:-N_g]

#        #Side contacts
#        mid_point_q2_index = \
#            (int)((q2_2D_in_3D_index_start + q2_2D_in_3D_index_end)/2)
#
##        residual_array[q3_2D_in_3D_index_start-N_g,
##                  mid_point_q2_index-5-N_g:mid_point_q2_index+5+1-N_g,
##                  :q1_2D_in_3D_index_start-N_g
##                 ] = \
##        phi_array[q3_2D_in_3D_index_start,
##                  mid_point_q2_index-5:mid_point_q2_index+5+1,
##                  N_g:q1_2D_in_3D_index_start
##                 ] - 0.1*0.
##
##        residual_array[q3_2D_in_3D_index_start-N_g,
##                  mid_point_q2_index-5-N_g:mid_point_q2_index+5+1-N_g,
##                  q1_2D_in_3D_index_end+1-N_g:
##                 ] = \
##        phi_array[q3_2D_in_3D_index_start,
##                  mid_point_q2_index-5:mid_point_q2_index+5+1,
##                  q1_2D_in_3D_index_end+1:-N_g
##                 ] + 0.1*0.
#
#        residual_array[q3_2D_in_3D_index_start-N_g,
#                       q2_2D_in_3D_index_start-N_g:q2_2D_in_3D_index_end+1-N_g,
#                      :q1_2D_in_3D_index_start-N_g
#                      ] = \
#            phi_array[q3_2D_in_3D_index_start,
#                      q2_2D_in_3D_index_start:q2_2D_in_3D_index_end+1,
#                      N_g:q1_2D_in_3D_index_start
#                     ] - 0.1*0.
#
#        residual_array[q3_2D_in_3D_index_start-N_g,
#                       q2_2D_in_3D_index_start-N_g:q2_2D_in_3D_index_end+1-N_g,
#                       q1_2D_in_3D_index_end+1-N_g:
#                      ] = \
#            phi_array[q3_2D_in_3D_index_start,
#                      q2_2D_in_3D_index_start:q2_2D_in_3D_index_end+1,
#                      q1_2D_in_3D_index_end+1:-N_g
#                     ] + 0.1*0.

        #Side contacts
        mid_point_q2_index = \
            (int)((q2_2D_in_3D_index_start + q2_2D_in_3D_index_end)/2)

        size_of_inflow   = 5.
        length_y         = self.obj.q2_end - self.obj.q2_start
        N_inflow_zones   = (int)(size_of_inflow/length_y*self.obj.N_q2)
        N_outflow_zones  = N_inflow_zones

        q2_inflow_start   = mid_point_q2_index-N_inflow_zones/2
        q2_inflow_end     = mid_point_q2_index+N_inflow_zones/2+1 

        q2_outflow_start  = mid_point_q2_index-N_outflow_zones/2
        q2_outflow_end    = mid_point_q2_index+N_outflow_zones/2+1

        residual_array[q3_2D_in_3D_index_start-N_g,
                       q2_inflow_start-N_g:q2_inflow_end-N_g,
                       :q1_2D_in_3D_index_start-N_g
                      ] = \
        phi_array[q3_2D_in_3D_index_start,
                  q2_inflow_start:q2_inflow_end,
                  N_g:q1_2D_in_3D_index_start
                 ] - 0.1*0.

        residual_array[q3_2D_in_3D_index_start-N_g,
                       q2_outflow_start-N_g:q2_outflow_end-N_g,
                       q1_2D_in_3D_index_end+1-N_g:
                      ] = \
        phi_array[q3_2D_in_3D_index_start,
                  q2_outflow_start:q2_outflow_end,
                  q1_2D_in_3D_index_end+1:-N_g
                 ] + 0.1*0.
        return

def compute_electrostatic_fields(self):
    
    if(self.performance_test_flag == True):
        tic = af.time()

    N_g = self.N_ghost

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
        phi_local_array.reshape([self.poisson.N_q3_3D_local+2*N_g,
                                 self.poisson.N_q2_3D_local+2*N_g,
                                 self.poisson.N_q1_3D_local+2*N_g
                                ]
                               )

    q3_2D_in_3D_index_start = self.poisson.q3_2D_in_3D_index_start 
    q3_2D_in_3D_index_end   = self.poisson.q3_2D_in_3D_index_end   
                                                           
    q1_2D_in_3D_index_start = self.poisson.q1_2D_in_3D_index_start 
    q1_2D_in_3D_index_end   = self.poisson.q1_2D_in_3D_index_end   
    q2_2D_in_3D_index_start = self.poisson.q2_2D_in_3D_index_start 
    q2_2D_in_3D_index_end   = self.poisson.q2_2D_in_3D_index_end   
    
    phi_2D_local_array = \
        phi_local_array[q3_2D_in_3D_index_start,
                        q2_2D_in_3D_index_start-N_g:q2_2D_in_3D_index_end+1+N_g,
                        q1_2D_in_3D_index_start-N_g:q1_2D_in_3D_index_end+1+N_g
                       ]
    
    phi_2D_local_array = \
        phi_2D_local_array.reshape([ (self.N_q1_local + 2*N_g) \
                                    *(self.N_q2_local + 2*N_g)
                                   ]
                                  )
    
    # convert from np->af
    self.phi = af.to_array(phi_2D_local_array) 
    # Since rho was defined at (i + 0.5, j + 0.5)
    # Electric Potential returned will also be at (i + 0.5, j + 0.5)
    self.phi = af.moddims(self.phi,
                          self.N_q1_local + 2*N_g,
                          self.N_q2_local + 2*N_g
                         )
    params.phi = self.phi

    # Obtaining the electric field values at (i+0.5, j+0.5):
    self.E1 = -(  af.shift(self.phi, -1, 0)
                - af.shift(self.phi,  1, 0)
               ) / (2 * self.dq1)

    self.E2 = -(  af.shift(self.phi, 0, -1)
                - af.shift(self.phi, 0,  1)
               ) / (2 * self.dq2)

    af.eval(self.E1, self.E2)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return


def fft_poisson(self):
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
        rho =   self.physical_system.params.charge_electron \
              * (self.compute_moments('density')[N_g:-N_g, N_g:-N_g])

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

        self.E1[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E1_hat))
        self.E2[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E2_hat))

        # Applying Periodic B.C's:
        self._communicate_fields()
        af.eval(self.E1, self.E2)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
