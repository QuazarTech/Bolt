
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import arrayfire as af
import numpy as np
import pylab as pl

pl.rcParams['figure.figsize']  = 20, 7.5
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

N_q1_density = 33
N_q2_density = 33

N_ghost = 1

q1_2D_start = -.5; q1_2D_end = 0.5
q2_2D_start = -.5; q2_2D_end = 0.5
location_in_q3 = 1.

q3_3D_start =  0.; q3_3D_end = 2.

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

dq1_3D = dq1_2D
dq2_3D = dq2_2D
dq3_3D = dq1_2D

length_multiples_q1 = 1
length_multiples_q2 = 1
N_q1_poisson = (2*length_multiples_q1+1)*N_q1_density
N_q2_poisson = (2*length_multiples_q2+1)*N_q2_density
N_q3_poisson = (int)((q3_3D_end - q3_3D_start) / dq1_3D)

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

glob_epsilon  = da_3D.createGlobalVec()
local_epsilon = da_3D.createLocalVec()

((i_q1_3D_start, i_q2_3D_start, i_q3_3D_start),
 (N_q1_3D_local, N_q2_3D_local, N_q3_3D_local)
) = \
    da_3D.getCorners()

i_q1_3D = ( (i_q1_3D_start + 0.5)
           + np.arange(-N_ghost, N_q1_3D_local + N_ghost)
          )

i_q2_3D = ( (i_q2_3D_start + 0.5)
           + np.arange(-N_ghost, N_q2_3D_local + N_ghost)
          )

i_q3_3D = ( (i_q3_3D_start + 0.5)
           + np.arange(-N_ghost, N_q3_3D_local + N_ghost)
          )

length_q1_2d = (q1_2D_end - q1_2D_start)
length_q2_2d = (q2_2D_end - q2_2D_start)

q1_3D =  q1_2D_start - length_multiples_q1*length_q1_2d + i_q1_3D * dq1_3D
q2_3D =  q2_2D_start - length_multiples_q2*length_q2_2d + i_q2_3D * dq2_3D
q3_3D =  q3_3D_start + i_q3_3D * dq3_3D

glob_density_array = glob_density.getArray(readonly=0)
glob_density_array = glob_density_array.reshape([N_q2_2D_local, \
                                                 N_q1_2D_local], \
                                               )
glob_density_array[:] = -335.

epsilon_array = local_epsilon.getArray(readonly=0)
epsilon_array = epsilon_array.reshape([N_q3_3D_local + 2*N_ghost, \
                                       N_q2_3D_local + 2*N_ghost, \
                                       N_q1_3D_local + 2*N_ghost
        			      ]
                                     )
epsilon_array[:] = 1.


print("rank = ", comm.rank)
print("N_q1_poisson = ", N_q1_poisson)
print("N_q2_poisson = ", N_q2_poisson)
print("N_q3_poisson = ", N_q3_poisson)

offset = 2*N_ghost-1
print("i_q1_local_3D = [", i_q1_3D_start, ",", i_q1_3D_start+N_q1_3D_local+offset, "]")
print("i_q2_local_3D = [", i_q2_3D_start, ",", i_q2_3D_start+N_q2_3D_local+offset, "]")
print("i_q3_local_3D = [", i_q3_3D_start, ",", i_q3_3D_start+N_q3_3D_local+offset, "]")
print(" ")
print("i_q1_local_2D = [", i_q1_2D_start, ",", i_q1_2D_start+N_q2_2D_local+offset, "]")
print("i_q2_local_2D = [", i_q2_2D_start, ",", i_q2_2D_start+N_q2_2D_local+offset, "]")

# Figure out the coordinates of the 3D phi cube of the current rank
print("q1_3D_start = ", q1_3D[N_ghost])
print("q2_3D_start = ", q2_3D[N_ghost])
print("q3_3D_start = ", q3_3D[N_ghost])
print(" ")
print("q1_2D_start = ", q1_2D[N_ghost])
print("q2_2D_start = ", q2_2D[N_ghost])

q3_2D_in_3D_index_start = np.where(q3_3D > location_in_q3 - dq3_3D)[0][0]
q3_2D_in_3D_index_end   = np.where(q3_3D < location_in_q3 + dq3_3D)[0][-1]

q1_2D_in_3D_index_start = np.where(abs(q1_3D - q1_2D[0+N_ghost] ) < 1e-10)[0][0]
q1_2D_in_3D_index_end   = np.where(abs(q1_3D - q1_2D[-1-N_ghost]) < 1e-10)[0][0]
q2_2D_in_3D_index_start = np.where(abs(q2_3D - q2_2D[0+N_ghost] ) < 1e-10)[0][0]
q2_2D_in_3D_index_end   = np.where(abs(q2_3D - q2_2D[-1-N_ghost]) < 1e-10)[0][0]

print("q1_2D_in_3D_index_start = ", q1_2D_in_3D_index_start, "q1_3D_start = ", q1_3D[q1_2D_in_3D_index_start])
print("q1_2D_in_3D_index_end   = ", q1_2D_in_3D_index_end, "q1_3D_end   = ", q1_3D[q1_2D_in_3D_index_end])
print("q2_2D_in_3D_index_start = ", q2_2D_in_3D_index_start, "q2_3D_start = ", q2_3D[q2_2D_in_3D_index_start])
print("q2_2D_in_3D_index_end   = ", q2_2D_in_3D_index_end, "q2_3D_end   = ", q2_3D[q2_2D_in_3D_index_end])
print("q3_2D_in_3D_index_start = ", q3_2D_in_3D_index_start, "q3_3D_start = ", q3_3D[q3_2D_in_3D_index_start])
print("q3_2D_in_3D_index_end   = ", q3_2D_in_3D_index_end, "q3_3D_end   = ", q3_3D[q3_2D_in_3D_index_end])

print(" ")
print("After ghost zone offset:")
print("q1_3D_start = ", q1_3D[q1_2D_in_3D_index_start])
print("q1_3D_end   = ", q1_3D[q1_2D_in_3D_index_end])
print("q2_3D_start = ", q2_3D[q2_2D_in_3D_index_start])
print("q2_3D_end   = ", q2_3D[q2_2D_in_3D_index_end])

epsilon_array[:q3_2D_in_3D_index_start,
              :, :
             ] = 10.
q3_3D_data_structure = 0.*epsilon_array
for j in range(q3_3D_data_structure.shape[1]):
    for i in range(q3_3D_data_structure.shape[2]):
        q3_3D_data_structure[:, j, i] = q3_3D

print("z = ", q3_3D_data_structure[q3_2D_in_3D_index_start, 0, 0])
class poisson_eqn(object):

    def __init__(self):
        self.local_phi = local_phi
        self.residual_counter = 0

    def compute_residual(self, snes, phi, residual):
        self.residual_counter += 1
#        print("residual iter = ", self.residual_counter)
        da_3D.globalToLocal(phi, local_phi)

        N_g = N_ghost

        phi_array = local_phi.getArray(readonly=0)
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

        phi_array[:N_ghost, :, :]               = 0.
        phi_array[N_q3_3D_local+N_ghost:, :, :] = 0.
        phi_array[:, :N_ghost, :]               = 0.
        phi_array[:, N_q2_3D_local+N_ghost:, :] = 0.
        phi_array[:, :, :N_ghost]               = 0.
        phi_array[:, :, N_q1_3D_local+N_ghost:] = 0.

        z = q3_3D_data_structure
        z_sample   = q3_3D[q3_2D_in_3D_index_start]
        z_backgate = q3_3D[0]
        side_wall_boundaries = (z_sample - z)/(z_sample - z_backgate)

        phi_array[:q3_2D_in_3D_index_start, :N_ghost, :]               = \
            side_wall_boundaries[:q3_2D_in_3D_index_start, :N_ghost, :]

        phi_array[:q3_2D_in_3D_index_start, N_q2_3D_local+N_ghost:, :] = \
            side_wall_boundaries[:q3_2D_in_3D_index_start, N_q2_3D_local+N_ghost:, :]

        phi_array[:q3_2D_in_3D_index_start, :, :N_ghost]               = \
            side_wall_boundaries[:q3_2D_in_3D_index_start, :, :N_ghost]

        phi_array[:q3_2D_in_3D_index_start, :, N_q1_3D_local+N_ghost:] = \
            side_wall_boundaries[:q3_2D_in_3D_index_start, :, N_q1_3D_local+N_ghost:]

        #Backgate
#        phi_array[:N_ghost,
#                  q2_2D_in_3D_index_start:q2_2D_in_3D_index_end+1,
#                  q1_2D_in_3D_index_start:q1_2D_in_3D_index_end+1
#                 ]  = 1.
        phi_array[:N_ghost, :, :]  = 1.

        phi_plus_x  = np.roll(phi_array, shift=-1, axis=2) # (i+3/2, j+1/2, k+1/2)
        phi_minus_x = np.roll(phi_array, shift=1,  axis=2) # (i-1/2, j+1/2, k+1/2)
        phi_plus_y  = np.roll(phi_array, shift=-1, axis=1) # (i+1/2, j+3/2, k+1/2)
        phi_minus_y = np.roll(phi_array, shift=1,  axis=1) # (i+1/2, j-1/2, k+1/2)
        phi_plus_z  = np.roll(phi_array, shift=-1, axis=0) # (i+1/2, j+1/2, k+3/2)
        phi_minus_z = np.roll(phi_array, shift=1,  axis=0) # (i+1/2, j+1/2, k+3/2)

        eps_left_edge  = epsilon_array # (i, j+1/2, k+1/2)
        eps_right_edge = np.roll(epsilon_array, shift=-1, axis=2) # (i+1, j+1/2, k+1/2)

        eps_bot_edge   = epsilon_array # (i+1/2, j, k+1/2)
        eps_top_edge   = np.roll(epsilon_array, shift=-1, axis=1) # (i+1/2, j+1, k+1/2)

        eps_back_edge  = epsilon_array # (i+1/2, j+1/2, k)
        eps_front_edge = np.roll(epsilon_array, shift=-1, axis=0) # (i+1/2, j+1/2, k+1)

        D_left_edge  = eps_left_edge  * (phi_array  - phi_minus_x)/dq1_3D
        D_right_edge = eps_right_edge * (phi_plus_x - phi_array)  /dq1_3D

        D_bot_edge   = eps_bot_edge   * (phi_array  - phi_minus_y)/dq2_3D
        D_top_edge   = eps_top_edge   * (phi_plus_y - phi_array  )/dq2_3D

        D_back_edge  = eps_back_edge  * (phi_array  - phi_minus_z)/dq3_3D
        D_front_edge = eps_front_edge * (phi_plus_z - phi_array  )/dq3_3D

        laplacian_phi =  (D_right_edge - D_left_edge) /dq1_3D \
                       + (D_top_edge   - D_bot_edge)  /dq2_3D \
                       + (D_front_edge - D_back_edge) /dq3_3D

#        d2phi_dx2   = (phi_minus_x - 2.*phi_array + phi_plus_x)/dq1_3D**2.
#        d2phi_dy2   = (phi_minus_y - 2.*phi_array + phi_plus_y)/dq2_3D**2.
#        d2phi_dz2   = (phi_minus_z - 2.*phi_array + phi_plus_z)/dq3_3D**2.
#        
#        laplacian_phi = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

        laplacian_phi[q3_2D_in_3D_index_start,
                      q2_2D_in_3D_index_start:q2_2D_in_3D_index_end+1,
                      q1_2D_in_3D_index_start:q1_2D_in_3D_index_end+1
                     ] \
                     += glob_density_array

        residual_array[:, :, :] = \
            laplacian_phi[N_g:-N_g, N_g:-N_g, N_g:-N_g]

        #Side contacts
        mid_point_q2_index = \
            (int)((q2_2D_in_3D_index_start + q2_2D_in_3D_index_end)/2)

        residual_array[q3_2D_in_3D_index_start-N_g,
                  mid_point_q2_index-5-N_g:mid_point_q2_index+5+1-N_g,
                  :q1_2D_in_3D_index_start-N_g
                 ] = \
        phi_array[q3_2D_in_3D_index_start,
                  mid_point_q2_index-5:mid_point_q2_index+5+1,
                  N_g:q1_2D_in_3D_index_start
                 ] - 0.1

        residual_array[q3_2D_in_3D_index_start-N_g,
                  mid_point_q2_index-5-N_g:mid_point_q2_index+5+1-N_g,
                  q1_2D_in_3D_index_end+1-N_g:
                 ] = \
        phi_array[q3_2D_in_3D_index_start,
                  mid_point_q2_index-5:mid_point_q2_index+5+1,
                  q1_2D_in_3D_index_end+1:-N_g
                 ] + 0.1*0.

        return

snes = PETSc.SNES().create()
pde  = poisson_eqn()
snes.setFunction(pde.compute_residual, glob_residual)

snes.setDM(da_3D)
snes.setFromOptions()

snes.solve(None, glob_phi)
phi_array = glob_phi.getArray()
phi_array = phi_array.reshape([N_q3_3D_local, \
                               N_q2_3D_local, \
                               N_q1_3D_local]
                             )
pl.subplot(121)
pl.contourf(
            phi_array[q3_2D_in_3D_index_start, :, :], 100, cmap='jet'
           )
pl.colorbar()
pl.title('Top View')
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.gca().set_aspect('equal')

pl.subplot(122)
pl.contourf(phi_array[:, N_q2_poisson/2, :], 100, cmap='jet')
pl.title('Side View')
pl.xlabel('$x$')
pl.ylabel('$z$')
pl.colorbar()
pl.gca().set_aspect('equal')
pl.show()

#for n in range(10):
#
#    print("====== n = ", n, "======")
#    glob_density_array[:] = n
#    snes.solve(None, glob_phi)
#    pde.residual_counter = 0
