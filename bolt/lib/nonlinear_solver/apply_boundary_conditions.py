import arrayfire as af
import numpy as np

def apply_bcs_f(self):
    
    if(self.performance_test_flag == True):
        tic = af.time()
    
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_ghost = self.N_ghost

    if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):

        A_q1 = self._convert_to_p_expanded(af.tile(self._A_q1, 
                                                   self.f.shape[0],
                                                   self.f.shape[1]
                                                  )
                                          )

        if(i_q1_start == 0):
    
            f_left = self.physical_system.boundary_conditions.\
                     f_left(self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.physical_system.params
                           )

            if(self.time_elapsed == 0):
                pass

            else:
                # Only changing outgoing characteristics:
                f_left = self._convert_to_p_expanded(f_left)
                self.f = self._convert_to_p_expanded(self.f)

                f_left = af.select(A_q1>0, f_left, self.f)

                f_left = self._convert_to_q_expanded(f_left)
                self.f = self._convert_to_q_expanded(self.f)

            self.f[:self.N_ghost] = f_left[:self.N_ghost]

        if(i_q1_end == self.N_q1 - 1):
            
            f_right = self.physical_system.boundary_conditions.\
                      f_right(self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.physical_system.params
                             )
  
            if(self.time_elapsed == 0):
                pass

            else:
                # Only changing outgoing characteristics:
                f_right = self._convert_to_p_expanded(f_right)
                self.f  = self._convert_to_p_expanded(self.f)

                f_right = af.select(A_q1<0, f_right, self.f)

                f_right = self._convert_to_q_expanded(f_right)
                self.f  = self._convert_to_q_expanded(self.f)

            self.f[-self.N_ghost:] = f_right[-self.N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q1 == 'mirror'):
        if(i_q1_start == 0):
            self.f[:N_ghost] = af.flip(self.f[N_ghost:2 * N_ghost], 0)
            self.f[:N_ghost] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    1
                                                   )
                                           )[:N_ghost]

        if(i_q1_end == self.N_q1 - 1):
            self.f[-N_ghost:] = af.flip(self.f[-2 * N_ghost:-N_ghost], 0)
            self.f[-N_ghost:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    1
                                                   )
                                           )[-N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q1 == 'shear'):
        # Not-implemented
        pass
        
    elif(self.physical_system.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):
    
        A_q2 = self._convert_to_p_expanded(af.tile(self._A_q2, 
                                                   self.f.shape[0],
                                                   self.f.shape[1]
                                                  )
                                          )

        if(i_q2_start == 0):
    
            f_bot = self.physical_system.boundary_conditions.\
                    f_bot(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            if(self.time_elapsed == 0):
                pass

            else:
                # Only changing outgoing characteristics:
                f_bot  = self._convert_to_p_expanded(f_bot)
                self.f = self._convert_to_p_expanded(self.f)

                f_bot = af.select(A_q2>0, f_bot, self.f)

                f_bot  = self._convert_to_q_expanded(f_bot)
                self.f = self._convert_to_q_expanded(self.f)

            self.f[:, :self.N_ghost] = f_bot[:, :self.N_ghost]

        if(i_q2_end == self.N_q2 - 1):
            
            f_top = self.physical_system.boundary_conditions.\
                    f_top(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )
  
            if(self.time_elapsed == 0):
                pass

            else:
                # Only changing outgoing characteristics:
                f_top  = self._convert_to_p_expanded(f_top)
                self.f = self._convert_to_p_expanded(self.f)

                f_top = af.select(A_q2<0, f_top, self.f)

                f_top  = self._convert_to_q_expanded(f_top)
                self.f = self._convert_to_q_expanded(self.f)

            self.f[:, -self.N_ghost:] = f_top[:, -self.N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q2 == 'mirror'):

        if(i_q2_start == 0):
            self.f[:, :N_ghost] = af.flip(self.f[:, N_ghost:2 * N_ghost], 1)
            self.f[:, :N_ghost] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    2
                                                   )
                                           )[:, :N_ghost]

        if(i_q2_end == self.N_q2 - 1):
            self.f[:, -N_ghost:] = af.flip(self.f[:, -2 * N_ghost:-N_ghost], 1)
            self.f[:, -N_ghost:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                    2
                                                   )
                                           )[:, -N_ghost:]

    elif(self.physical_system.boundary_conditions.in_q2 == 'shear'):
        # Not implemented
        pass

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
    
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()

    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_ghost = self.N_ghost

    if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):

        if(i_q1_start == 0):

            self.E1[:N_ghost] = self.physical_system.boundary.E1_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )
            self.E2[:N_ghost] = self.physical_system.boundary.E2_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )
            self.E3[:N_ghost] = self.physical_system.boundary.E3_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )
            
            self.B1[:N_ghost] = self.physical_system.boundary.B1_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )
            self.B2[:N_ghost] = self.physical_system.boundary.B2_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )
            self.B3[:N_ghost] = self.physical_system.boundary.B3_left(self.q1, self.q2,
                                                                      self.p1, self.p2, self.p3, 
                                                                      self.physical_system.params
                                                                     )

        if(i_q1_end == self.N_q1 - 1):

            self.E1[-N_ghost:] = self.physical_system.boundary.E1_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.E2[-N_ghost:] = self.physical_system.boundary.E2_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.E3[-N_ghost:] = self.physical_system.boundary.E3_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            
            self.B1[-N_ghost:] = self.physical_system.boundary.B1_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.B2[-N_ghost:] = self.physical_system.boundary.B2_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.B3[-N_ghost:] = self.physical_system.boundary.B3_right(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )

    elif(self.physical_system.boundary_conditions.in_q1 == 'mirror'):

        if(i_q1_start == 0):
            
            self.E1[:N_ghost] = af.flip(self.E1[N_ghost:2 * N_ghost], 0)
            self.E2[:N_ghost] = af.flip(self.E2[N_ghost:2 * N_ghost], 0)
            self.E3[:N_ghost] = af.flip(self.E3[N_ghost:2 * N_ghost], 0)
            
            self.B1[:N_ghost] = af.flip(self.B1[N_ghost:2 * N_ghost], 0)
            self.B2[:N_ghost] = af.flip(self.B2[N_ghost:2 * N_ghost], 0)
            self.B3[:N_ghost] = af.flip(self.B3[N_ghost:2 * N_ghost], 0)

        if(i_q1_end == self.N_q1 - 1):

            self.E1[-N_ghost:] = af.flip(self.E1[-2 * N_ghost:-N_ghost], 0)
            self.E2[-N_ghost:] = af.flip(self.E2[-2 * N_ghost:-N_ghost], 0)
            self.E3[-N_ghost:] = af.flip(self.E3[-2 * N_ghost:-N_ghost], 0)

            self.B1[-N_ghost:] = af.flip(self.B1[-2 * N_ghost:-N_ghost], 0)
            self.B2[-N_ghost:] = af.flip(self.B2[-2 * N_ghost:-N_ghost], 0)
            self.B3[-N_ghost:] = af.flip(self.B3[-2 * N_ghost:-N_ghost], 0)

    elif(self.physical_system.boundary_conditions.in_q1 == 'shear'):
        # Not-implemented
        pass
        
    elif(self.physical_system.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):

        if(i_q2_start == 0):
            
            self.E1[:, :N_ghost] = self.physical_system.boundary.E1_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.E2[:, :N_ghost] = self.physical_system.boundary.E2_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.E3[:, :N_ghost] = self.physical_system.boundary.E3_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            
            self.B1[:, :N_ghost] = self.physical_system.boundary.B1_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.B2[:, :N_ghost] = self.physical_system.boundary.B2_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )
            self.B3[:, :N_ghost] = self.physical_system.boundary.B3_bot(self.q1, self.q2,
                                                                        self.p1, self.p2, self.p3, 
                                                                        self.physical_system.params
                                                                       )

        if(i_q2_end == self.N_q2 - 1):
            
            self.E1[:, -N_ghost:] = self.physical_system.boundary.E1_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )
            self.E2[:, -N_ghost:] = self.physical_system.boundary.E2_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )
            self.E3[:, -N_ghost:] = self.physical_system.boundary.E3_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )
            
            self.B1[:, -N_ghost:] = self.physical_system.boundary.B1_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )
            self.B2[:, -N_ghost:] = self.physical_system.boundary.B2_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )
            self.B3[:, -N_ghost:] = self.physical_system.boundary.B3_top(self.q1, self.q2,
                                                                         self.p1, self.p2, self.p3, 
                                                                         self.physical_system.params
                                                                        )

    elif(self.physical_system.boundary_conditions.in_q2 == 'mirror'):

        if(i_q2_start == 0):
            
            self.E1[:, :N_ghost] = af.flip(self.E1[:, N_ghost:2 * N_ghost], 1)
            self.E2[:, :N_ghost] = af.flip(self.E2[:, N_ghost:2 * N_ghost], 1)
            self.E3[:, :N_ghost] = af.flip(self.E3[:, N_ghost:2 * N_ghost], 1)
            
            self.B1[:, :N_ghost] = af.flip(self.B1[:, N_ghost:2 * N_ghost], 1)
            self.B2[:, :N_ghost] = af.flip(self.B2[:, N_ghost:2 * N_ghost], 1)
            self.B3[:, :N_ghost] = af.flip(self.B3[:, N_ghost:2 * N_ghost], 1)

        if(i_q2_end == self.N_q2 - 1):
            
            self.E1[:, -N_ghost:] = af.flip(self.E1[:, -2 * N_ghost:-N_ghost], 1)
            self.E2[:, -N_ghost:] = af.flip(self.E2[:, -2 * N_ghost:-N_ghost], 1)
            self.E3[:, -N_ghost:] = af.flip(self.E3[:, -2 * N_ghost:-N_ghost], 1)

            self.B1[:, -N_ghost:] = af.flip(self.B1[:, -2 * N_ghost:-N_ghost], 1)
            self.B2[:, -N_ghost:] = af.flip(self.B2[:, -2 * N_ghost:-N_ghost], 1)
            self.B3[:, -N_ghost:] = af.flip(self.B3[:, -2 * N_ghost:-N_ghost], 1)

    elif(self.physical_system.boundary_conditions.in_q2 == 'shear'):
        # Not implemented
        pass

    elif(self.physical_system.boundary_conditions.in_q2 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    af.eval(self.E1, self.E2, self.E3, self.B1, self.B2, self.B3)
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_fields += toc - tic

    self._communicate_distribution_function()
    
    return
