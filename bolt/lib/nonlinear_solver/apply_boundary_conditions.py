#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def apply_bcs_f(self):

    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_g = self.N_ghost

    if(self.boundary_conditions.in_q1 == 'dirichlet'):
        # A_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        A_q1 = af.tile(self._A_q1, 1,
                       self.f.shape[1],
                       self.f.shape[2]
                      )

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            
            f_left = self.boundary_conditions.\
                     f_left(self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.physical_system.params
                           )

            # Only changing inflowing characteristics:
            f_left = af.select(A_q1>0, f_left, self.f)

            self.f[:, :N_g] = f_left[:, :N_g]

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            
            f_right = self.boundary_conditions.\
                      f_right(self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.physical_system.params
                             )

            # Only changing inflowing characteristics:
            f_right = af.select(A_q1<0, f_right, self.f)

            self.f[:, -N_g:] = f_right[:, -N_g:]

    elif(self.boundary_conditions.in_q1 == 'mirror'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;
            self.f[:, :N_g] = af.flip(self.f[:, N_g:2 * N_g], 1)
            
            self.f[:N_g] = af.flip(self.f[N_g:2 * N_g], 0)
            
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f[:, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    0
                                                   )
                                           )[:, :N_g]

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            
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

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.boundary_conditions.in_q2 == 'dirichlet'):

        # _A_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        
        A_q2 = af.tile(self._A_q2, 1, 
                       self.f.shape[1],
                       self.f.shape[2]
                      )

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
    
            f_bot = self.boundary_conditions.\
                    f_bot(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            # Only changing inflowing characteristics:
            f_bot = af.select(A_q2>0, f_bot, self.f)

            self.f[:, :, :N_g] = f_bot[:, :, :N_g]

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            f_top = self.boundary_conditions.\
                    f_top(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            # Only changing inflowing characteristics:
            f_top = af.select(A_q2<0, f_top, self.f)

            self.f[:, :, -N_g:] = f_top[:, :, -N_g:]

    elif(self.boundary_conditions.in_q2 == 'mirror'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):

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

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
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

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.boundary_conditions.in_q2 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

def apply_bcs_fields(self):

    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_g = self.N_ghost

    if(self.boundary_conditions.in_q1 == 'dirichlet'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):

            E1 = self.boundary_conditions.\
                 E1_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]
            E2 = self.boundary_conditions.\
                 E2_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]
            E3 = self.boundary_conditions.\
                 E3_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]
            
            B1 = self.boundary_conditions.\
                 B1_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]
            B2 = self.boundary_conditions.\
                 B2_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]
            B3 = self.boundary_conditions.\
                 B3_left(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :N_g]

            self.cell_centered_EM_fields[:, :N_g] = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):

            E1 = self.boundary_conditions.\
                 E1_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]
            E2 = self.boundary_conditions.\
                 E2_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]
            E3 = self.boundary_conditions.\
                 E3_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]
            
            B1 = self.boundary_conditions.\
                 B1_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]
            B2 = self.boundary_conditions.\
                 B2_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]
            B3 = self.boundary_conditions.\
                 B3_right(self.q1_center, self.q2_center,
                          self.physical_system.params
                         )[:, -N_g:]

            self.cell_centered_EM_fields[:, -N_g:] = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

    elif(self.boundary_conditions.in_q1 == 'mirror'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;

            self.cell_centered_EM_fields[:, :N_g] = \
                af.flip(self.cell_centered_EM_fields[:, N_g:2 * N_g], 1)

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;
            
            self.cell_centered_EM_fields[:, -N_g:] = \
                af.flip(self.cell_centered_EM_fields[:, -2 * N_g:-N_g], 1)

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.boundary_conditions.in_q2 == 'dirichlet'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
            
            E1 = self.boundary_conditions.\
                 E1_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]
            E2 = self.boundary_conditions.\
                 E2_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]
            E3 = self.boundary_conditions.\
                 E3_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]
           
            B1 = self.boundary_conditions.\
                 B1_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]
            B2 = self.boundary_conditions.\
                 B2_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]
            B3 = self.boundary_conditions.\
                 B3_bot(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, :N_g]

            self.cell_centered_EM_fields[:, :, :N_g] = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            E1 = self.boundary_conditions.\
                  E1_top(self.q1_center, self.q2_center,
                         self.physical_system.params
                        )[:, :, -N_g:]
            E2 = self.boundary_conditions.\
                 E2_top(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, -N_g:]
            E3 = self.boundary_conditions.\
                 E3_top(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, -N_g:]
            
            B1 = self.boundary_conditions.\
                 B1_top(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, -N_g:]
            B2 = self.boundary_conditions.\
                 B2_top(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, -N_g:]
            B3 = self.boundary_conditions.\
                 B3_top(self.q1_center, self.q2_center,
                        self.physical_system.params
                       )[:, :, -N_g:]
            
            self.cell_centered_EM_fields[:, :, -N_g:] = \
                af.join(0, E1, E2, E3, af.join(0, B1, B2, B3))

    elif(self.boundary_conditions.in_q2 == 'mirror'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
            
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;

            self.cell_centered_EM_fields[:, :, :N_g] = \
                af.flip(self.cell_centered_EM_fields[:, :, N_g:2 * N_g], 2)

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;

            self.cell_centered_EM_fields[:, :, -N_g:] = \
                af.flip(self.cell_centered_EM_fields[:, :, -2 * N_g:-N_g], 2)

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.boundary_conditions.in_q2 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    af.eval(self.cell_centered_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_fields += toc - tic

    return