#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def apply_dirichlet_bcs_f(self, boundary):
    
    N_g = self.N_ghost
    
    if(self._A_q1.elements() == self.N_p1 * self.N_p2 * self.N_p3):
        # A_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        A_q1 = af.tile(self._A_q1, 1,
                       self.f.shape[1],
                       self.f.shape[2]
                      )

    if(self._A_q2.elements() == self.N_p1 * self.N_p2 * self.N_p3):
        # _A_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        A_q2 = af.tile(self._A_q2, 1, 
                       self.f.shape[1],
                       self.f.shape[2]
                      )

    if(boundary == 'left'):
        f_left = self.boundary_conditions.\
                 f_left(self.f, self.q1_center, self.q2_center,
                        self.p1, self.p2, self.p3, 
                        self.physical_system.params
                       )

        # NOTE: This is not necessary since the Riemann solver ensures that the
        # information outflow characteristics in the ghost zones do not affect
        # the numerical solution inside in the physical domain.
        #
        # Only changing inflowing characteristics:
        #f_left = af.select(A_q1>0, f_left, self.f)

        self.f[:, :N_g] = f_left[:, :N_g]

    elif(boundary == 'right'):
        f_right = self.boundary_conditions.\
                  f_right(self.f, self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

        # Only changing inflowing characteristics:
        #f_right = af.select(A_q1<0, f_right, self.f)

        self.f[:, -N_g:] = f_right[:, -N_g:]

    elif(boundary == 'bottom'):
        f_bottom = self.boundary_conditions.\
                   f_bottom(self.f, self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.physical_system.params
                           )

        # Only changing inflowing characteristics:
        #f_bottom = af.select(A_q2>0, f_bottom, self.f)

        self.f[:, :, :N_g] = f_bottom[:, :, :N_g]

    elif(boundary == 'top'):
        f_top = self.boundary_conditions.\
                f_top(self.f, self.q1_center, self.q2_center,
                      self.p1, self.p2, self.p3, 
                      self.physical_system.params
                     )

        # Only changing inflowing characteristics:
        #f_top = af.select(A_q2<0, f_top, self.f)

        self.f[:, :, -N_g:] = f_top[:, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f(self, boundary):

    N_g = self.N_ghost

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :N_g] = af.flip(self.f[:, N_g:2 * N_g], 1)
        
        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, :N_g] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, :N_g]

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, -N_g:] = af.flip(self.f[:, -2 * N_g:-N_g], 1)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, -N_g:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, -N_g:]

    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :N_g] = af.flip(self.f[:, :, N_g:2 * N_g], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, :N_g] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, :N_g]

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, -N_g:] = af.flip(self.f[:, :, -2 * N_g:-N_g], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, -N_g:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, -N_g:]

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
            try:
                assert(self.boundary_conditions.in_q1_right == 'periodic')
            except:
                raise Exception('Periodic boundary conditions need to be applied to \
                                 both the boundaries of a particular axis'
                               )
            pass

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
            try:
                assert(self.boundary_conditions.in_q2_top == 'periodic')
            except:
                raise Exception('Periodic boundary conditions need to be applied to \
                                 both the boundaries of a particular axis'
                               )
            pass

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

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

def apply_dirichlet_bcs_fields(self, boundary):
    
#    N_g = self.N_ghost
#
#    # These arguments are defined since they are required by all the function calls:
#    # So the functions can be called instead using function(*args)
#    args = (self.q1_center, self.q2_center, self.physical_system.params)
#    
#    if(boundary == 'left'):
#        E1 = self.boundary_conditions.\
#             E1_left(self.cell_centered_EM_fields[0],*args)[:, :N_g]
#
#        E2 = self.boundary_conditions.\
#             E2_left(self.cell_centered_EM_fields[1],*args)[:, :N_g]
#
#        E3 = self.boundary_conditions.\
#             E3_left(self.cell_centered_EM_fields[2],*args)[:, :N_g]
#        
#        B1 = self.boundary_conditions.\
#             B1_left(self.cell_centered_EM_fields[3],*args)[:, :N_g]
#
#        B2 = self.boundary_conditions.\
#             B2_left(self.cell_centered_EM_fields[4],*args)[:, :N_g]
#
#        B3 = self.boundary_conditions.\
#             B3_left(self.cell_centered_EM_fields[5],*args)[:, :N_g]
#
#        self.cell_centered_EM_fields[:, :N_g] = \
#                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
#
#    elif(boundary == 'right'):
#        E1 = self.boundary_conditions.\
#             E1_right(self.cell_centered_EM_fields[0],*args)[:, -N_g:]
#        
#        E2 = self.boundary_conditions.\
#             E2_right(self.cell_centered_EM_fields[1],*args)[:, -N_g:]
#
#        E3 = self.boundary_conditions.\
#             E3_right(self.cell_centered_EM_fields[2],*args)[:, -N_g:]
#        
#        B1 = self.boundary_conditions.\
#             B1_right(self.cell_centered_EM_fields[3],*args)[:, -N_g:]
#        
#        B2 = self.boundary_conditions.\
#             B2_right(self.cell_centered_EM_fields[4],*args)[:, -N_g:]
#        
#        B3 = self.boundary_conditions.\
#             B3_right(self.cell_centered_EM_fields[5],*args)[:, -N_g:]
#
#        self.cell_centered_EM_fields[:, -N_g:] = \
#            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
#
#    elif(boundary == 'bottom'):
#        E1 = self.boundary_conditions.\
#             E1_bottom(self.cell_centered_EM_fields[0],*args)[:, :, :N_g]
#
#        E2 = self.boundary_conditions.\
#             E2_bottom(self.cell_centered_EM_fields[1],*args)[:, :, :N_g]
#        
#        E3 = self.boundary_conditions.\
#             E3_bottom(self.cell_centered_EM_fields[2],*args)[:, :, :N_g]
#       
#        B1 = self.boundary_conditions.\
#             B1_bottom(self.cell_centered_EM_fields[3],*args)[:, :, :N_g]
#
#        B2 = self.boundary_conditions.\
#             B2_bottom(self.cell_centered_EM_fields[4],*args)[:, :, :N_g]
#
#        B3 = self.boundary_conditions.\
#             B3_bottom(self.cell_centered_EM_fields[5],*args)[:, :, :N_g]
#
#        self.cell_centered_EM_fields[:, :, :N_g] = \
#            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
#
#    elif(boundary == 'top'):
#        E1 = self.boundary_conditions.\
#             E1_top(self.cell_centered_EM_fields[0],*args)[:, :, -N_g:]
#
#        E2 = self.boundary_conditions.\
#             E2_top(self.cell_centered_EM_fields[1],*args)[:, :, -N_g:]
#
#        E3 = self.boundary_conditions.\
#             E3_top(self.cell_centered_EM_fields[2],*args)[:, :, -N_g:]
#        
#        B1 = self.boundary_conditions.\
#             B1_top(self.cell_centered_EM_fields[3],*args)[:, :, -N_g:]
#
#        B2 = self.boundary_conditions.\
#             B2_top(self.cell_centered_EM_fields[4],*args)[:, :, -N_g:]
#
#        B3 = self.boundary_conditions.\
#             B3_top(self.cell_centered_EM_fields[5],*args)[:, :, -N_g:]
#        
#        self.cell_centered_EM_fields[:, :, -N_g:] = \
#            af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))
#    
#    else:
#        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_fields(self, boundary):
    
#    N_g = self.N_ghost
#
#    if(boundary == 'left'):
#        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
#        #   0   1   2   3   4   5
#        # For mirror boundary conditions:
#        # 0 = 5; 1 = 4; 2 = 3;
#        self.cell_centered_EM_fields[:, :N_g] = \
#            af.flip(self.cell_centered_EM_fields[:, N_g:2 * N_g], 1)
#
#    elif(boundary == 'right'):
#        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
#        #      -6  -5  -4  -3  -2  -1
#        # For mirror boundary conditions:
#        # -1 = -6; -2 = -5; -3 = -4;
#        
#        self.cell_centered_EM_fields[:, -N_g:] = \
#            af.flip(self.cell_centered_EM_fields[:, -2 * N_g:-N_g], 1)
#
#    elif(boundary == 'bottom'):
#        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
#        #   0   1   2   3   4   5
#        # For mirror boundary conditions:
#        # 0 = 5; 1 = 4; 2 = 3;
#
#        self.cell_centered_EM_fields[:, :, :N_g] = \
#            af.flip(self.cell_centered_EM_fields[:, :, N_g:2 * N_g], 2)
#
#    elif(boundary == 'top'):
#        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
#        #      -6  -5  -4  -3  -2  -1
#        # For mirror boundary conditions:
#        # -1 = -6; -2 = -5; -3 = -4;
#
#        self.cell_centered_EM_fields[:, :, -N_g:] = \
#            af.flip(self.cell_centered_EM_fields[:, :, -2 * N_g:-N_g], 2)
#
#    else:
#        raise Exception('Invalid choice for boundary')

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
            try:
                assert(self.boundary_conditions.in_q1_right == 'periodic')
            except:
                raise Exception('Periodic boundary conditions need to be applied to \
                                 both the boundaries of a particular axis'
                               )
            pass

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
            try:
                assert(self.boundary_conditions.in_q2_top == 'periodic')
            except:
                raise Exception('Periodic boundary conditions need to be applied to \
                                 both the boundaries of a particular axis'
                               )
            pass

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

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.cell_centered_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_fields += toc - tic

    return
