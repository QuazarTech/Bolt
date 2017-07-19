#!/usr/bin/env python 
# -*- coding: utf-8 -*-

def fdtd(self, dt):
    
  # E's and B's are staggered in time such that
  # B's are defined at (n + 1/2), and E's are defined at n 
  
  # Positions of grid point where field quantities are defined:
  # B_x --> (i, j + 1/2)
  # B_y --> (i + 1/2, j)
  # B_z --> (i + 1/2, j + 1/2)
  
  # E_x --> (i + 1/2, j)
  # E_y --> (i, j + 1/2)
  # E_z --> (i, j)
  
  # J_x --> (i + 1/2, j)
  # J_y --> (i, j + 1/2)
  # J_z --> (i, j)

  # The communicate function transfers the data from the local vectors to the global
  # vectors, in addition to dealing with the boundary conditions:
  self._communicate_fields()

  # dEx/dt = + dBz/dy
  # dEy/dt = - dBz/dx
  # dEz/dt = dBy/dx - dBx/dy
  args.E_x +=   (dt/dy) * (args.B_z - af.shift(args.B_z, 1, 0)) - args.J_x * dt
  args.E_y +=  -(dt/dx) * (args.B_z - af.shift(args.B_z, 0, 1)) - args.J_y * dt
  args.E_z +=   (dt/dx) * (args.B_y - af.shift(args.B_y, 0, 1)) \
              - (dt/dy) * (args.B_x - af.shift(args.B_x, 1, 0)) \
              - dt * args.J_z
          
  # Applying boundary conditions:
  args = non_linear_solver.communicate.communicate_fields(da, args, local, glob)

  # dBx/dt = -dEz/dy
  # dBy/dt = +dEz/dx
  # dBz/dt = - ( dEy/dx - dEx/dy )
  args.B_x +=  -(dt/dy) * (af.shift(args.E_z, -1, 0) - args.E_z)
  args.B_y +=   (dt/dx) * (af.shift(args.E_z, 0, -1) - args.E_z)
  args.B_z += - (dt/dx) * (af.shift(args.E_y, 0, -1) - args.E_y) \
              + (dt/dy) * (af.shift(args.E_x, -1, 0) - args.E_x)

  # Applying boundary conditions:
  args = non_linear_solver.communicate.communicate_fields(da, args, local, glob)

  return(args)

def fdtd_grid_to_ck_grid(E_x, E_y, E_z, B_x, B_y, B_z):

  # Interpolating at the (i + 1/2, j + 1/2) point of the grid to use for the CK solver:    
  E_x = 0.5 * (E_x + af.shift(E_x, -1, 0)) #(i + 1/2, j + 1/2)
  B_x = 0.5 * (B_x + af.shift(B_x, 0, -1)) #(i + 1/2, j + 1/2)

  E_y = 0.5 * (E_y + af.shift(E_y, 0, -1)) #(i + 1/2, j + 1/2)
  B_y = 0.5 * (B_y + af.shift(B_y, -1, 0)) #(i + 1/2, j + 1/2)

  E_z = 0.25 * (
                E_z + af.shift(E_z, 0, -1) + \
                af.shift(E_z, -1, 0) + af.shift(E_z, -1, -1)
               ) #(i + 1/2, j + 1/2)

  af.eval(E_x, E_y, E_z, B_x, B_y, B_z)
  return(E_x, E_y, E_z, B_x, B_y, B_z)