import arrayfire as af
import numpy as np

def apply_bcs_f(self):
    
    if(self.performance_test_flag == True):
        tic = af.time()
    
    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_ghost = self.N_ghost

    if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):
        # _A_q1 is of shape (1, 1, Np1 * Np2 * Np3)
        # We tile to get it to form (Nq1, Nq2, Np1 * Np2 * Np3)
        A_q1 = af.tile(self._A_q1, 
                       self.f.shape[0],
                       self.f.shape[1]
                      )

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            
            f_left = self.physical_system.boundary_conditions.\
                     f_left(self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.physical_system.params
                           )

            # Only changing inflowing characteristics:
            f_left = af.select(A_q1>0, f_left, self.f)

            self.f[:N_ghost] = f_left[:N_ghost]

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            
            f_right = self.physical_system.boundary_conditions.\
                      f_right(self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.physical_system.params
                             )

            # Only changing inflowing characteristics:
            f_right = af.select(A_q1<0, f_right, self.f)

            self.f[-N_ghost:] = f_right[-N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q1 == 'mirror'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;
            self.f[:N_ghost] = af.flip(self.f[N_ghost:2 * N_ghost], 0)
            
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f[:N_ghost] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    1
                                                   )
                                           )[:N_ghost]

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;
            self.f[-N_ghost:] = af.flip(self.f[-2 * N_ghost:-N_ghost], 0)

            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f[-N_ghost:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    1
                                                   )
                                           )[-N_ghost:]

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.physical_system.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):

        # _A_q2 is of shape (1, 1, Np1 * Np2 * Np3)
        # We tile to get it to form (Nq1, Nq2, Np1 * Np2 * Np3)
        A_q2 = af.tile(self._A_q2, 
                       self.f.shape[0],
                       self.f.shape[1]
                      )

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
    
            f_bot = self.physical_system.boundary_conditions.\
                    f_bot(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            # Only changing inflowing characteristics:
            f_bot = af.select(A_q2>0, f_bot, self.f)

            self.f[:, :N_ghost] = f_bot[:, :N_ghost]

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            f_top = self.physical_system.boundary_conditions.\
                    f_top(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            # Only changing inflowing characteristics:
            f_top = af.select(A_q2<0, f_top, self.f)

            self.f[:, -N_ghost:] = f_top[:, -N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q2 == 'mirror'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;
            self.f[:, :N_ghost] = af.flip(self.f[:, N_ghost:2 * N_ghost], 1)

            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f[:, :N_ghost] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    2
                                                   )
                                           )[:, :N_ghost]

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;
            self.f[:, -N_ghost:] = af.flip(self.f[:, -2 * N_ghost:-N_ghost], 1)

            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f[:, -N_ghost:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    2
                                                   )
                                           )[:, -N_ghost:]

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.physical_system.boundary_conditions.in_q2 == 'periodic'):
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
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_ghost = self.N_ghost

    if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):

            self.E1[:N_ghost] = self.physical_system.boundary.E1_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]
            self.E2[:N_ghost] = self.physical_system.boundary.E2_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]
            self.E3[:N_ghost] = self.physical_system.boundary.E3_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]
            
            self.B1[:N_ghost] = self.physical_system.boundary.B1_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]
            self.B2[:N_ghost] = self.physical_system.boundary.B2_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]
            self.B3[:N_ghost] = self.physical_system.boundary.B3_left(self.q1, self.q2,
                                                                      self.physical_system.params
                                                                     )[:N_ghost]

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):

            self.E1[-N_ghost:] = self.physical_system.boundary.E1_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]
            self.E2[-N_ghost:] = self.physical_system.boundary.E2_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]
            self.E3[-N_ghost:] = self.physical_system.boundary.E3_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]
            
            self.B1[-N_ghost:] = self.physical_system.boundary.B1_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]
            self.B2[-N_ghost:] = self.physical_system.boundary.B2_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]
            self.B3[-N_ghost:] = self.physical_system.boundary.B3_right(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[-N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q1 == 'mirror'):

        # If local zone includes the left physical boundary:
        if(i_q1_start == 0):
            
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;

            self.E1[:N_ghost] = af.flip(self.E1[N_ghost:2 * N_ghost], 0)
            self.E2[:N_ghost] = af.flip(self.E2[N_ghost:2 * N_ghost], 0)
            self.E3[:N_ghost] = af.flip(self.E3[N_ghost:2 * N_ghost], 0)
            
            self.B1[:N_ghost] = af.flip(self.B1[N_ghost:2 * N_ghost], 0)
            self.B2[:N_ghost] = af.flip(self.B2[N_ghost:2 * N_ghost], 0)
            self.B3[:N_ghost] = af.flip(self.B3[N_ghost:2 * N_ghost], 0)

        # If local zone includes the right physical boundary:
        if(i_q1_end == self.N_q1 - 1):
            
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;
            
            self.E1[-N_ghost:] = af.flip(self.E1[-2 * N_ghost:-N_ghost], 0)
            self.E2[-N_ghost:] = af.flip(self.E2[-2 * N_ghost:-N_ghost], 0)
            self.E3[-N_ghost:] = af.flip(self.E3[-2 * N_ghost:-N_ghost], 0)

            self.B1[-N_ghost:] = af.flip(self.B1[-2 * N_ghost:-N_ghost], 0)
            self.B2[-N_ghost:] = af.flip(self.B2[-2 * N_ghost:-N_ghost], 0)
            self.B3[-N_ghost:] = af.flip(self.B3[-2 * N_ghost:-N_ghost], 0)

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.physical_system.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
            
            self.E1[:, :N_ghost] = self.physical_system.boundary.E1_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]
            self.E2[:, :N_ghost] = self.physical_system.boundary.E2_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]
            self.E3[:, :N_ghost] = self.physical_system.boundary.E3_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]
            
            self.B1[:, :N_ghost] = self.physical_system.boundary.B1_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]
            self.B2[:, :N_ghost] = self.physical_system.boundary.B2_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]
            self.B3[:, :N_ghost] = self.physical_system.boundary.B3_bot(self.q1, self.q2,
                                                                        self.physical_system.params
                                                                       )[:, :N_ghost]

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            self.E1[:, -N_ghost:] = self.physical_system.boundary.E1_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]
            self.E2[:, -N_ghost:] = self.physical_system.boundary.E2_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]
            self.E3[:, -N_ghost:] = self.physical_system.boundary.E3_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]
            
            self.B1[:, -N_ghost:] = self.physical_system.boundary.B1_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]
            self.B2[:, -N_ghost:] = self.physical_system.boundary.B2_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]
            self.B3[:, -N_ghost:] = self.physical_system.boundary.B3_top(self.q1, self.q2,
                                                                         self.physical_system.params
                                                                        )[:, -N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q2 == 'mirror'):

        # If local zone includes the bottom physical boundary:
        if(i_q2_start == 0):
            
            # x-0-x-0-x-0-|-0-x-0-x-0-x-....
            #   0   1   2   3   4   5
            # For mirror boundary conditions:
            # 0 = 5; 1 = 4; 2 = 3;

            self.E1[:, :N_ghost] = af.flip(self.E1[:, N_ghost:2 * N_ghost], 1)
            self.E2[:, :N_ghost] = af.flip(self.E2[:, N_ghost:2 * N_ghost], 1)
            self.E3[:, :N_ghost] = af.flip(self.E3[:, N_ghost:2 * N_ghost], 1)
            
            self.B1[:, :N_ghost] = af.flip(self.B1[:, N_ghost:2 * N_ghost], 1)
            self.B2[:, :N_ghost] = af.flip(self.B2[:, N_ghost:2 * N_ghost], 1)
            self.B3[:, :N_ghost] = af.flip(self.B3[:, N_ghost:2 * N_ghost], 1)

        # If local zone includes the top physical boundary:
        if(i_q2_end == self.N_q2 - 1):
            
            # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
            #      -6  -5  -4  -3  -2  -1
            # For mirror boundary conditions:
            # -1 = -6; -2 = -5; -3 = -4;

            self.E1[:, -N_ghost:] = af.flip(self.E1[:, -2 * N_ghost:-N_ghost], 1)
            self.E2[:, -N_ghost:] = af.flip(self.E2[:, -2 * N_ghost:-N_ghost], 1)
            self.E3[:, -N_ghost:] = af.flip(self.E3[:, -2 * N_ghost:-N_ghost], 1)

            self.B1[:, -N_ghost:] = af.flip(self.B1[:, -2 * N_ghost:-N_ghost], 1)
            self.B2[:, -N_ghost:] = af.flip(self.B2[:, -2 * N_ghost:-N_ghost], 1)
            self.B3[:, -N_ghost:] = af.flip(self.B3[:, -2 * N_ghost:-N_ghost], 1)

    # This is automatically handled by the PETSc function globalToLocal()
    elif(self.physical_system.boundary_conditions.in_q2 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    af.eval(self.E1, self.E2, self.E3, self.B1, self.B2, self.B3)
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_fields += toc - tic

    return
