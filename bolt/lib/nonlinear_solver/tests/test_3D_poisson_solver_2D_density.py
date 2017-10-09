
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import arrayfire as af
import numpy as np
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

comm = PETSc.COMM_WORLD.tompi4py()

N_q1_poisson = 70
N_q2_poisson = 70
N_q3_poisson = 70

N_q1_density = 35
N_q2_density = 35

N_ghost = 3

da_3D = PETSc.DMDA().create([N_q1_poisson,
                             N_q2_poisson,
                             N_q3_poisson],
                             stencil_width = N_ghost,
                             boundary_type = ('periodic',
                                              'periodic',
                                              'periodic'
                                             ),
                              stencil_type = 1,
                              dof          = 1,
                              comm         = comm
                           )

glob_phi      = da_3D.createGlobalVec()
local_phi     = da_3D.createLocalVec()
glob_residual = da_3D.createGlobalVec()

((i_q1_3D_start, i_q2_3D_start, i_q3_3D_start),
 (N_q1_3D_local, N_q2_3D_local, N_q3_3D_local)
) = \
    da_3D.getCorners()

q1_3D_start = -2.; q1_3D_end = 2.
q2_3D_start = -2.; q2_3D_end = 2.
q3_3D_start =  0.; q3_3D_end = 2.

dq1_3D = (q1_3D_end - q1_3D_start) / N_q1_poisson
dq2_3D = (q2_3D_end - q2_3D_start) / N_q2_poisson
dq3_3D = (q3_3D_end - q3_3D_start) / N_q3_poisson

i_q1_3D = ( (i_q1_3D_start + 0.5)
           + np.arange(-N_ghost, N_q1_3D_local + N_ghost)
          )

i_q2_3D = ( (i_q2_3D_start + 0.5)
           + np.arange(-N_ghost, N_q2_3D_local + N_ghost)
          )

i_q3_3D = ( (i_q3_3D_start + 0.5)
           + np.arange(-N_ghost, N_q3_3D_local + N_ghost)
          )

q1_3D =  q1_3D_start + i_q1_3D * dq1_3D
q2_3D =  q2_3D_start + i_q2_3D * dq2_3D
q3_3D =  q3_3D_start + i_q3_3D * dq3_3D

da_2D = PETSc.DMDA().create([N_q1_density,
                             N_q2_density],
                             stencil_width = N_ghost,
                             boundary_type = ('periodic',
                                              'periodic'
                                             ),
                             stencil_type = 1,
                             dof          = 1,
                             comm         = comm
                           )

glob_density  = da_2D.createGlobalVec()
local_density = da_2D.createLocalVec()


((i_q1_2D_start, i_q2_2D_start), 
 (N_q1_2D_local, N_q2_2D_local)
) = \
    da_2D.getCorners()

q1_2D_start = -1.; q1_2D_end = 1.
q2_2D_start = -1.; q2_2D_end = 1.
location_in_q3 = 1.

dq1_2D = (q1_2D_end - q1_2D_start) / N_q1_density
dq2_2D = (q2_2D_end - q2_2D_start) / N_q2_density

i_q1_2D = ( (i_q1_2D_start + 0.5)
           + np.arange(-N_ghost, N_q1_2D_local + N_ghost)
          )

i_q2_2D = ( (i_q2_2D_start + 0.5)
           + np.arange(-N_ghost, N_q2_2D_local + N_ghost)
          )

q1_2D =  q1_2D_start + i_q1_2D * dq1_2D
q2_2D =  q2_2D_start + i_q2_2D * dq2_2D

glob_density_array = glob_density.getArray(readonly=0)
glob_density_array = glob_density_array.reshape([N_q2_2D_local, \
                                                 N_q1_2D_local, 1], \
                                               )
glob_density_array[:] = 1.

da_2D.globalToLocal(glob_density, local_density)

density_array = local_density.getArray(readonly=0)
density_array = density_array.reshape([N_q2_2D_local + 2*N_ghost, \
                                       N_q1_2D_local + 2*N_ghost, 1], \
                                     )

print("rank = ", comm.rank)

# Figure out the coordinates of the 3D phi cube of the current rank
print("q1_3D_start = ", q1_3D[N_ghost])
print("q2_3D_start = ", q2_3D[N_ghost])
print("q3_3D_start = ", q3_3D[N_ghost])
print(" ")
print("q1_2D_start = ", q1_2D[N_ghost])
print("q2_2D_start = ", q2_2D[N_ghost])

q1_2D_in_3D_index_start = np.where(q1_3D > q1_2D[0]  - dq1_3D)[0][0]
q1_2D_in_3D_index_end   = np.where(q1_3D < q1_2D[-1] + dq1_3D)[0][-1]
q2_2D_in_3D_index_start = np.where(q2_3D > q2_2D[0]  - dq2_3D)[0][0]
q2_2D_in_3D_index_end   = np.where(q2_3D < q2_2D[-1] + dq2_3D)[0][-1]
q3_2D_in_3D_index_start = np.where(q3_3D > location_in_q3 - dq3_3D)[0][0]
q3_2D_in_3D_index_end   = np.where(q3_3D < location_in_q3 + dq3_3D)[0][-1]

print("q1_2D_in_3D_index_start = ", q1_2D_in_3D_index_start, "q1_3D_start = ", q1_3D[q1_2D_in_3D_index_start])
print("q1_2D_in_3D_index_end   = ", q1_2D_in_3D_index_end, "q1_3D_end   = ", q1_3D[q1_2D_in_3D_index_end])
print("q2_2D_in_3D_index_start = ", q2_2D_in_3D_index_start, "q2_3D_start = ", q2_3D[q2_2D_in_3D_index_start])
print("q2_2D_in_3D_index_end   = ", q2_2D_in_3D_index_end, "q2_3D_end   = ", q2_3D[q2_2D_in_3D_index_end])
print("q3_2D_in_3D_index_start = ", q3_2D_in_3D_index_start, "q3_3D_start = ", q3_3D[q3_2D_in_3D_index_start])
print("q3_2D_in_3D_index_end   = ", q3_2D_in_3D_index_end, "q3_3D_end   = ", q3_3D[q3_2D_in_3D_index_end])

class poisson_eqn(object):

    def __init__(self):
        self.local_phi = local_phi

    def compute_residual(self, snes, phi, residual):
        da_3D.globalToLocal(phi, local_phi)

        N_g = N_ghost

        phi_array = local_phi.getArray(readonly=0)
        phi_array = phi_array.reshape([N_q3_3D_local + 2*N_g, \
                                       N_q2_3D_local + 2*N_g, \
                                       N_q1_3D_local + 2*N_g, 1
        			      ]
                                     )
    
        residual_array = residual.getArray(readonly=0)
        residual_array = residual_array.reshape([N_q3_3D_local, \
                                                 N_q2_3D_local, \
                                                 N_q1_3D_local, 1
        					]
                                               )

        phi_array[:N_ghost, :, :]               = 0.
        phi_array[N_q1_3D_local+N_ghost:, :, :] = 0.
        phi_array[:, :N_ghost, :]               = 0.
        phi_array[:, N_q2_3D_local+N_ghost:, :] = 0.
        phi_array[:, :, :N_ghost]               = 0.
        phi_array[:, :, N_q3_3D_local+N_ghost:] = 0.

        phi_plus_x  = np.roll(phi_array, shift=-1, axis=2)
        phi_minus_x = np.roll(phi_array, shift=1,  axis=2)
        phi_plus_y  = np.roll(phi_array, shift=-1, axis=1)
        phi_minus_y = np.roll(phi_array, shift=1,  axis=1)
        phi_plus_z  = np.roll(phi_array, shift=-1, axis=0)
        phi_minus_z = np.roll(phi_array, shift=1,  axis=0)

        d2phi_dx2   = (phi_minus_x - 2.*phi_array + phi_plus_x)/dq1_3D**2.
        d2phi_dy2   = (phi_minus_y - 2.*phi_array + phi_plus_y)/dq2_3D**2.
        d2phi_dz2   = (phi_minus_z - 2.*phi_array + phi_plus_z)/dq3_3D**2.
        
        laplacian_phi = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

        laplacian_phi[q3_2D_in_3D_index_start:q3_2D_in_3D_index_end,
                      q2_2D_in_3D_index_start:q2_2D_in_3D_index_end,
                      q1_2D_in_3D_index_start:q1_2D_in_3D_index_end
                     ] \
                     += density_array

        residual_array[:, :, :] = \
            laplacian_phi[N_g:-N_g, N_g:-N_g, N_g:-N_g]

        return

snes = PETSc.SNES().create()
pde  = poisson_eqn()
snes.setFunction(pde.compute_residual, glob_residual)

snes.setDM(da_3D)
snes.setFromOptions()
snes.solve(None, glob_phi)
