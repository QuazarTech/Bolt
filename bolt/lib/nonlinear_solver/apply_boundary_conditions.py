#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import time

def apply_shearing_box_bcs_f(self, boundary):
    
    N_g_q = self.N_ghost_q
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :N_g_q] - q * omega * L_q1 * self.time_elapsed
        
        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)

        self.f[:, :N_g_q] = af.reorder(af.approx2(af.reorder(self.f[:, :N_g_q], 1, 2, 0),
                                                  af.reorder(self.q1_center[:, :N_g_q], 1, 2, 0),
                                                  af.reorder(sheared_coordinates, 1, 2, 0),
                                                  af.INTERP.BICUBIC_SPLINE,
                                                  xp = af.reorder(self.q1_center[:, :N_g_q], 1, 2, 0),
                                                  yp = af.reorder(self.q2_center[:, :N_g_q], 1, 2, 0)
                                                 ),
                                       2, 0, 1
                                      )
        
    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, -N_g_q:] + q * omega * L_q1 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)

        self.f[:, -N_g_q:] = af.reorder(af.approx2(af.reorder(self.f[:, -N_g_q:], 1, 2, 0),
                                                   af.reorder(self.q1_center[:, -N_g_q:], 1, 2, 0),
                                                   af.reorder(sheared_coordinates, 1, 2, 0),
                                                   af.INTERP.BICUBIC_SPLINE,
                                                   xp = af.reorder(self.q1_center[:, -N_g_q:], 1, 2, 0),
                                                   yp = af.reorder(self.q2_center[:, -N_g_q:], 1, 2, 0)
                                                  ),
                                        2, 0, 1
                                       )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :N_g_q] - q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.f[:, :, :N_g_q] = af.reorder(af.approx2(af.reorder(self.f[:, :, :N_g_q], 1, 2, 0),
                                                     af.reorder(sheared_coordinates, 1, 2, 0),
                                                     af.reorder(self.q2_center[:, :, :N_g_q], 1, 2, 0),
                                                     af.INTERP.BICUBIC_SPLINE,
                                                     xp = af.reorder(self.q1_center[:, :, :N_g_q], 1, 2, 0),
                                                     yp = af.reorder(self.q2_center[:, :, :N_g_q], 1, 2, 0)
                                                    ),
                                          2, 0, 1
                                         )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, -N_g_q:] + q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )
        
        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.f[:, :, -N_g_q:] = af.reorder(af.approx2(af.reorder(self.f[:, :, -N_g_q:], 1, 2, 0),
                                                    af.reorder(sheared_coordinates, 1, 2, 0),
                                                    af.reorder(self.q2_center[:, :, -N_g_q:], 1, 2, 0),
                                                    af.INTERP.BICUBIC_SPLINE,
                                                    xp = af.reorder(self.q1_center[:, :, -N_g_q:], 1, 2, 0),
                                                    yp = af.reorder(self.q2_center[:, :, -N_g_q:], 1, 2, 0)
                                                   ),
                                           2, 0, 1
                                          )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_f(self, boundary):
    
    N_g_q = self.N_ghost_q
    N_s   = self.N_species

    # Number of DOF in the array for a single species:
    dof = self.dof

    for i in range(N_s):

        A_q1 = self._A_q(self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params, i
                        )[0]

        A_q2 = self._A_q(self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params, i
                        )[1]

        if(A_q1.elements() == self.N_species * self.dof):
            # If A_q1 is of shape (Np1 * Np2 * Np3)
            # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
            A_q1 = af.tile(A_q1, 1,
                           self.f.shape[1],
                           self.f.shape[2]
                          )

        if(A_q2.elements() == self.N_species * self.dof):
            # If A_q2 is of shape (Np1 * Np2 * Np3)
            # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
            A_q2 = af.tile(A_q2, 1, 
                           self.f.shape[1],
                           self.f.shape[2]
                          )
    
        if(boundary == 'left'):
            f_left = list(map(lambda f:f[:, :N_g_q],
                              self.boundary_conditions.\
                              f_left(self.f, self.q1_center, self.q2_center,
                                     self.p1_center, self.p2_center, self.p3_center, 
                                     self.physical_system.params
                                    )
                             )
                         )[i]

            # Only changing inflowing characteristics:
            self.f[i * dof:(i+1) * dof, :N_g_q] = \
                af.select(A_q1>0, f_left, 
                          self.f[i * dof:(i+1) * dof, :N_g_q]
                         )

        elif(boundary == 'right'):
            f_right = list(map(lambda f:f[:, -N_g_q:],
                               self.boundary_conditions.\
                               f_right(self.f, self.q1_center, self.q2_center,
                                       self.p1_center, self.p2_center, self.p3_center, 
                                       self.physical_system.params
                                      )
                              )
                          )[i]

            # Only changing inflowing characteristics:
            self.f[i * dof:(i+1) * dof, -N_g_q:] = \
                af.select(A_q1<0, f_right, 
                          self.f[i * dof:(i+1) * dof, -N_g_q:]
                         )

        elif(boundary == 'bottom'):
            f_bottom = list(map(lambda f:f[:, :, :N_g_q],
                                self.boundary_conditions.\
                                f_bottom(self.f, self.q1_center, self.q2_center,
                                         self.p1_center, self.p2_center, self.p3_center, 
                                         self.physical_system.params
                                        )
                               )
                           )[i]

            # Only changing inflowing characteristics:
            self.f[i * dof:(i+1) * dof, :, :N_g_q] = \
                af.select(A_q2>0, f_bottom[i], 
                          self.f[i * dof:(i+1) * dof, :, :N_g_q]
                         )

        elif(boundary == 'top'):
            f_top = list(map(lambda f:f[:, :, -N_g_q:],
                             self.boundary_conditions.\
                             f_top(self.f, self.q1_center, self.q2_center,
                                   self.p1_center, self.p2_center, self.p3_center, 
                                   self.physical_system.params
                                  )
                            )
                        )[i]
    
            # Only changing inflowing characteristics:
            self.f[i * dof:(i+1) * dof, :, -N_g_q:] = \
                af.select(A_q2<0, f_top[i], 
                          self.f[i * dof:(i+1) * dof, :, -N_g_q:]
                         )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f(self, boundary):

    N_g_q = self.N_ghost_q
    dof   = self.dof 

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :N_g_q] = af.flip(self.f[:, N_g_q:2 * N_g_q], 1)
        
        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        for i in range(self.N_species):
            self.f[i * dof:(i+1) * dof, :N_g_q] = \
                self._convert_to_q_expanded(af.flip(self.\
                                                    _convert_to_p_expanded(self.f[i * dof:
                                                                                  (i+1) * dof
                                                                                 ]
                                                                          ), 
                                                    0
                                                   )
                                           )[:, :N_g_q]

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, -N_g_q:] = af.flip(self.f[:, -2 * N_g_q:-N_g_q], 1)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        for i in range(self.N_species):
            self.f[i * dof:(i+1) * dof, -N_g_q:] = \
                self._convert_to_q_expanded(af.flip(self.\
                                                    _convert_to_p_expanded(self.f[i * dof:
                                                                                  (i+1) * dof
                                                                                 ]
                                                                          ), 
                                                    0
                                                   )
                                           )[:, -N_g_q:]
        
    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :N_g_q] = af.flip(self.f[:, :, N_g_q:2 * N_g_q], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        for i in range(self.N_species):
            self.f[i * dof:(i+1) * dof, :, :N_g_q] = \
                self._convert_to_q_expanded(af.flip(self.\
                                                    _convert_to_p_expanded(self.f[i * dof:
                                                                                  (i+1) * dof
                                                                                 ]
                                                                          ), 
                                                    1
                                                   )
                                           )[:, :, :N_g_q]

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, -N_g_q:] = af.flip(self.f[:, :, -2 * N_g_q:-N_g_q], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        for i in range(self.N_species):
            self.f[i * dof:(i+1) * dof, :, -N_g_q:] = \
                self._convert_to_q_expanded(af.flip(self.\
                                                    _convert_to_p_expanded(self.f[i * dof:
                                                                                  (i+1) * dof
                                                                                 ]
                                                                          ), 
                                                    1
                                                   )
                                           )[:, :, -N_g_q:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_bcs_f(self):
    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    # If local zone includes the left physical boundary:
    if(i_q1_start == 0):

        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'left')

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            apply_mirror_bcs_f(self, 'left')            

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'left')            
            apply_dirichlet_bcs_f(self, 'left')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_left == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'left')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')
    
    # If local zone includes the right physical boundary:
    if(i_q1_end == self.N_q1 - 1):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'right')

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            apply_mirror_bcs_f(self, 'right')
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'right')            
            apply_dirichlet_bcs_f(self, 'right')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_right == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'right')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == 0):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'bottom')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            apply_mirror_bcs_f(self, 'bottom')            

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'bottom')            
            apply_dirichlet_bcs_f(self, 'bottom')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_bottom == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'bottom')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.N_q2 - 1):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'top')

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            apply_mirror_bcs_f(self, 'top')
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'top')            
            apply_dirichlet_bcs_f(self, 'top')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_top == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'top')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

def apply_shearing_box_bcs_fields(self, boundary):
    
    N_g_q = self.N_ghost_q
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :N_g_q] - q * omega * L_q1 * self.time_elapsed
        
        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.cell_centered_EM_fields[:, :N_g_q] = \
            af.reorder(af.approx2(af.reorder(self.cell_centered_EM_fields[:, :N_g_q], 1, 2, 0),
                                  af.reorder(self.q1_center[:, :N_g_q], 1, 2, 0),
                                  af.reorder(sheared_coordinates, 1, 2, 0),
                                  af.INTERP.BICUBIC_SPLINE,
                                  xp = af.reorder(self.q1_center[:, :N_g_q], 1, 2, 0),
                                  yp = af.reorder(self.q2_center[:, :N_g_q], 1, 2, 0)
                                 ),
                       2, 0, 1
                      )
        
    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, -N_g_q:] + q * omega * L_q1 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.cell_centered_EM_fields[:, -N_g_q:] = \
            af.reorder(af.approx2(af.reorder(self.cell_centered_EM_fields[:, -N_g_q:], 1, 2, 0),
                                  af.reorder(self.q1_center[:, :N_g_q], 1, 2, 0),
                                  af.reorder(sheared_coordinates, 1, 2, 0),
                                  af.INTERP.BICUBIC_SPLINE,
                                  xp = af.reorder(self.q1_center[:, -N_g_q:], 1, 2, 0),
                                  yp = af.reorder(self.q2_center[:, -N_g_q:], 1, 2, 0)
                                 ),
                       2, 0, 1
                      )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :N_g_q] - q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )

        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.cell_centered_EM_fields[:, :, :N_g_q] = \
            af.reorder(af.approx2(af.reorder(self.cell_centered_EM_fields[:, :, :N_g_q], 1, 2, 0),
                                  af.reorder(sheared_coordinates, 1, 2, 0),
                                  af.reorder(self.q2_center[:, :, :N_g_q], 1, 2, 0),
                                  af.INTERP.BICUBIC_SPLINE,
                                  xp = af.reorder(self.q1_center[:, :, :N_g_q], 1, 2, 0),
                                  yp = af.reorder(self.q2_center[:, :, :N_g_q], 1, 2, 0)
                                 ),
                       2, 0, 1
                      )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, -N_g_q:] + q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )
        
        # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
        # and reordering back from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
        self.cell_centered_EM_fields[:, :, -N_g_q:] = \
            af.reorder(af.approx2(af.reorder(self.cell_centered_EM_fields[:, :, -N_g_q:], 1, 2, 0),
                                  af.reorder(sheared_coordinates, 1, 2, 0),
                                  af.reorder(self.q2_center[:, :, -N_g_q:], 1, 2, 0),
                                  af.INTERP.BICUBIC_SPLINE,
                                  xp = af.reorder(self.q1_center[:, :, -N_g_q:], 1, 2, 0),
                                  yp = af.reorder(self.q2_center[:, :, -N_g_q:], 1, 2, 0)
                                 ),
                       2, 0, 1
                      )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_fields(self, boundary):
    
    N_g_q = self.N_ghost_q

    # These arguments are defined since they are required by all the function calls:
    # So the functions can be called instead using function(*args)
    args = (self.q1_center, self.q2_center, self.physical_system.params)
    
    if(boundary == 'left'):
        E1 = self.boundary_conditions.\
             E1_left(self.cell_centered_EM_fields[0],*args)[:, :N_g_q]

        E2 = self.boundary_conditions.\
             E2_left(self.cell_centered_EM_fields[1],*args)[:, :N_g_q]

        E3 = self.boundary_conditions.\
             E3_left(self.cell_centered_EM_fields[2],*args)[:, :N_g_q]
        
        B1 = self.boundary_conditions.\
             B1_left(self.cell_centered_EM_fields[3],*args)[:, :N_g_q]

        B2 = self.boundary_conditions.\
             B2_left(self.cell_centered_EM_fields[4],*args)[:, :N_g_q]

        B3 = self.boundary_conditions.\
             B3_left(self.cell_centered_EM_fields[5],*args)[:, :N_g_q]

        self.cell_centered_EM_fields[:, :N_g_q] = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

    elif(boundary == 'right'):
        E1 = self.boundary_conditions.\
             E1_right(self.cell_centered_EM_fields[0],*args)[:, -N_g_q:]
        
        E2 = self.boundary_conditions.\
             E2_right(self.cell_centered_EM_fields[1],*args)[:, -N_g_q:]

        E3 = self.boundary_conditions.\
             E3_right(self.cell_centered_EM_fields[2],*args)[:, -N_g_q:]
        
        B1 = self.boundary_conditions.\
             B1_right(self.cell_centered_EM_fields[3],*args)[:, -N_g_q:]
        
        B2 = self.boundary_conditions.\
             B2_right(self.cell_centered_EM_fields[4],*args)[:, -N_g_q:]
        
        B3 = self.boundary_conditions.\
             B3_right(self.cell_centered_EM_fields[5],*args)[:, -N_g_q:]

        self.cell_centered_EM_fields[:, -N_g_q:] = \
            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

    elif(boundary == 'bottom'):
        E1 = self.boundary_conditions.\
             E1_bottom(self.cell_centered_EM_fields[0],*args)[:, :, :N_g_q]

        E2 = self.boundary_conditions.\
             E2_bottom(self.cell_centered_EM_fields[1],*args)[:, :, :N_g_q]
        
        E3 = self.boundary_conditions.\
             E3_bottom(self.cell_centered_EM_fields[2],*args)[:, :, :N_g_q]
       
        B1 = self.boundary_conditions.\
             B1_bottom(self.cell_centered_EM_fields[3],*args)[:, :, :N_g_q]

        B2 = self.boundary_conditions.\
             B2_bottom(self.cell_centered_EM_fields[4],*args)[:, :, :N_g_q]

        B3 = self.boundary_conditions.\
             B3_bottom(self.cell_centered_EM_fields[5],*args)[:, :, :N_g_q]

        self.cell_centered_EM_fields[:, :, :N_g_q] = \
            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

    elif(boundary == 'top'):
        E1 = self.boundary_conditions.\
             E1_top(self.cell_centered_EM_fields[0],*args)[:, :, -N_g_q:]

        E2 = self.boundary_conditions.\
             E2_top(self.cell_centered_EM_fields[1],*args)[:, :, -N_g_q:]

        E3 = self.boundary_conditions.\
             E3_top(self.cell_centered_EM_fields[2],*args)[:, :, -N_g_q:]
        
        B1 = self.boundary_conditions.\
             B1_top(self.cell_centered_EM_fields[3],*args)[:, :, -N_g_q:]

        B2 = self.boundary_conditions.\
             B2_top(self.cell_centered_EM_fields[4],*args)[:, :, -N_g_q:]

        B3 = self.boundary_conditions.\
             B3_top(self.cell_centered_EM_fields[5],*args)[:, :, -N_g_q:]
        
        self.cell_centered_EM_fields[:, :, -N_g_q:] = \
            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
    
    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_fields(self, boundary):
    
    N_g_q = self.N_ghost_q

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.cell_centered_EM_fields[:, :N_g_q] = \
            af.flip(self.cell_centered_EM_fields[:, N_g_q:2 * N_g_q], 1)

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        
        self.cell_centered_EM_fields[:, -N_g_q:] = \
            af.flip(self.cell_centered_EM_fields[:, -2 * N_g_q:-N_g_q], 1)

    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;

        self.cell_centered_EM_fields[:, :, :N_g_q] = \
            af.flip(self.cell_centered_EM_fields[:, :, N_g_q:2 * N_g_q], 2)

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;

        self.cell_centered_EM_fields[:, :, -N_g_q:] = \
            af.flip(self.cell_centered_EM_fields[:, :, -2 * N_g_q:-N_g_q], 2)

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_bcs_fields(self):

    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    # If local zone includes the left physical boundary:
    if(i_q1_start == 0):

        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
            apply_dirichlet_bcs_fields(self, 'left')

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            apply_mirror_bcs_fields(self, 'left')            

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            apply_mirror_bcs_fields(self, 'left')            
            apply_dirichlet_bcs_fields(self, 'left')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_left == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_fields(self, 'left')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')
    
    # If local zone includes the right physical boundary:
    if(i_q1_end == self.N_q1 - 1):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_fields(self, 'right')

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            apply_mirror_bcs_fields(self, 'right')
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            apply_mirror_bcs_fields(self, 'right')            
            apply_dirichlet_bcs_fields(self, 'right')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_right == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_fields(self, 'right')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == 0):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_fields(self, 'bottom')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            apply_mirror_bcs_fields(self, 'bottom')            

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            apply_mirror_bcs_fields(self, 'bottom')            
            apply_dirichlet_bcs_fields(self, 'bottom')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_bottom == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_fields(self, 'bottom')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.N_q2 - 1):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_fields(self, 'top')

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            apply_mirror_bcs_fields(self, 'top')
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            apply_mirror_bcs_fields(self, 'top')            
            apply_dirichlet_bcs_fields(self, 'top')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_top == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_fields(self, 'top')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.cell_centered_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_fields += toc - tic

    return
