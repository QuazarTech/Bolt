#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import arrayfire as af
import numpy as np
from numpy.fft import fftfreq
import pylab as pl

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

class poisson_eqn(object):
    """
    This user class is an application context for the problem at hand;
    It contains some parametes and frames the matrix system depending on
    the system state. The Poisson2D object is used by the
    compute_electrostatic_fields function in computing the electrostatic fields
    using the PETSc's KSP solver methods
    """

    def __init__(self, nonlinear_solver_obj):
        self.da        = nonlinear_solver_obj._da_snes
        self.obj       = nonlinear_solver_obj
        self.local_phi = self.da.createLocalVec() # phi with ghost zones
        self.N_ghost   = self.obj.N_ghost
        self.dq1       = self.obj.dq1
        self.dq2       = self.obj.dq2
        self.density   = 0.

        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = \
                self.da.getCorners()
        
        self.N_q1_local = N_q1_local
        self.N_q2_local = N_q2_local

    def compute_residual(self, snes, phi, residual):

        self.da.globalToLocal(phi, self.local_phi)

        N_g = self.N_ghost

	# Residual assembly using numpy
        phi_array = self.local_phi.getArray(readonly=0)
        phi_array = phi_array.reshape([self.N_q2_local + 2*N_g, \
                                       self.N_q1_local + 2*N_g, 1], \
                                      order='A'
                                     )

        residual_array = residual.getArray(readonly=0)
        residual_array = residual_array.reshape([self.N_q2_local, \
                                                 self.N_q1_local, 1], \
                                                order='A'
                                               )

        #phi_array[:N_g,                 :] = 0.
        #phi_array[self.N_q2_local+N_g:, :] = 0.
        #phi_array[:,                 :N_g] = 0.
        #phi_array[:, self.N_q1_local+N_g:] = 0.

        phi_plus_x  = np.roll(phi_array, shift=-1, axis=1)
        phi_minus_x = np.roll(phi_array, shift=1,  axis=1)
        phi_plus_y  = np.roll(phi_array, shift=-1, axis=0)
        phi_minus_y = np.roll(phi_array, shift=1,  axis=0)

        d2phi_dx2   = (phi_minus_x - 2.*phi_array + phi_plus_x)/self.dq1**2.
        d2phi_dy2   = (phi_minus_y - 2.*phi_array + phi_plus_y)/self.dq2**2.
	
        laplacian_phi = d2phi_dx2 + d2phi_dy2

        density_af = af.moddims(self.density,
                                  (self.N_q1_local+2*N_g)
                                * (self.N_q2_local+2*N_g)
                               )
        density_np = density_af.to_ndarray()
        density_np = density_np.reshape([self.N_q2_local + 2*N_g, \
                                         self.N_q1_local + 2*N_g, 1], \
                                        order='A'
                                       )

        residual_array[:, :] = \
            (laplacian_phi + density_np)[N_g:-N_g, N_g:-N_g]

#        residual_array[:, :] = \
#            (laplacian_phi + self.density)[N_g:-N_g, N_g:-N_g]


	# Residual assembly using arrayfire
#        phi_array    = self.local_phi.getArray(readonly=1)
#        phi_af_array = af.to_array(phi_array)
#        phi_af_array = af.moddims(phi_af_array,
#                                  self.N_q1_local + 2*N_g,
#                                  self.N_q2_local + 2*N_g
#                                 )
#
#        residual_af_array = phi_af_array[N_g:-N_g, N_g:-N_g]**2. - 2.
#        residual_af_array = af.moddims(residual_af_array,
#                                         self.N_q1_local
#                                       * self.N_q2_local
#                                      )
#        residual_array    = residual.getArray(readonly=0)
#        residual_array[:] = residual_af_array.to_ndarray()

        return

def compute_electrostatic_fields(self, performance_test_flag = False):
    
    if(performance_test_flag == True):
        tic = af.time()

    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_snes.getCorners()
    
    N_g = self.N_ghost

    snes = PETSc.SNES().create()
    pde  = poisson_eqn(self)
    snes.setFunction(pde.compute_residual, self.glob_residual)
    
    snes.setDM(self._da_snes)
    snes.setFromOptions()
    pde.density = self.compute_moments('density')
    snes.solve(None, self.glob_phi)
    
    #phi_array = self.glob_phi.getArray()
    #print("phi = ", phi_array)
    #phi_array = phi_array.reshape([N_q2_local, \
    #                               N_q1_local, 1], \
    #                              order='A'
    #                             )

    self._da_snes.globalToLocal(self.glob_phi, pde.local_phi)
    phi_local_array = pde.local_phi.getArray()
    electric_potential = af.to_array(phi_local_array)
    phi_local_array = phi_local_array.reshape([N_q2_local + 2*N_g, \
                                               N_q1_local + 2*N_g], \
                                             )
    density_af = af.moddims(pde.density,
                              (N_q1_local+2*N_g)
                            * (N_q2_local+2*N_g)
                           )
    density_np = density_af.to_ndarray()
    density_np = density_np.reshape([N_q2_local + 2*N_g, \
                                     N_q1_local + 2*N_g], \
                                   )
    # Since rho was defined at (i + 0.5, j + 0.5)
    # Electric Potential returned will also be at (i + 0.5, j + 0.5)
#    electric_potential = af.to_array(np.swapaxes(phi_local_array,
#                                                 0, 1
#                                                )
#                                    )

    electric_potential = af.moddims(electric_potential,
                                    N_q1_local + 2*N_g,
                                    N_q2_local + 2*N_g
                                   )

    # Obtaining the values at (i+0.5, j+0.5):
    self.E1 = -(  af.shift(electric_potential, -1)
                - af.shift(electric_potential,  1)
               ) / (2 * self.dq1)

    self.E2 = -(  af.shift(electric_potential, 0, -1)
                - af.shift(electric_potential, 0,  1)
               ) / (2 * self.dq2)

    af.eval(self.E1, self.E2)

    q2_minus = 0.25
    q2_plus  = 0.75

    E2_expected = -0.5/20 * (  af.log(af.cosh(( self.q2 - q2_minus)*20)) 
                             - af.log(af.cosh(( self.q2 - q2_plus )*20))
                            ) 

    pl.subplot(121)
    pl.contourf(
                #np.array(self.E2)[N_g:-N_g, N_g:-N_g], 100
                density_np[N_g:-N_g, N_g:-N_g], 100
               )
    pl.colorbar()
    #pl.axis('equal')
    pl.title(r'Density')
    #pl.title(r'$E^2_{numerical}$')
    pl.subplot(122)
    pl.contourf(
                #np.array(E2_expected)[N_g:-N_g, N_g:-N_g], 100
                phi_local_array[N_g:-N_g, N_g:-N_g], 100
               )
    pl.colorbar()
    pl.title(r'$\phi$')
    #pl.title(r'$E^2_{analytic}$')
    #pl.axis('equal')
    pl.show()


    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return


def fft_poisson(self, performance_test_flag = False):
    """
    Solves the Poisson Equation using the FFTs:

    Used as a backup solver in case of low resolution runs
    (ie. used on a single node) with periodic boundary
    conditions.
    """
    if(performance_test_flag == True):
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

        E1_hat = -1j * 2 * np.pi * (k_q1) * potential_hat
        E2_hat = -1j * 2 * np.pi * (k_q2) * potential_hat

        self.E1[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E1_hat))
        self.E2[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E2_hat))

        # Applying Periodic B.C's:
        self._communicate_fields(performance_test_flag)
        af.eval(self.E1, self.E2)

    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
