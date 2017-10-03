import arrayfire as af

def apply_bcs_f(self):
    
    if(self.performance_test_flag == True):
        tic = af.time()
    
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    N_ghost = self.N_ghost

    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    addition = lambda a, b:a + b

    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    q1_center_new = af.broadcast(addition, self.q1_center, - self._A_q1 * dt)
    q2_center_new = af.broadcast(addition, self.q2_center, - self._A_q2 * dt)

    if(self.physical_system.boundary_conditions.in_q1 == 'dirichlet'):

        if(i_q1_start == 0):
            f_left = self.physical_system.boundary_conditions.\
                     f_left(self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.physical_system.params
                           )

            self.f = af.select(self.q1_center_new < self.q1_start,
                               f_left,
                               self.f
                              )

        if(i_q1_end == self.N_q1 - 1):
            
            f_right = self.physical_system.boundary_conditions.\
                      f_right(self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.physical_system.params
                             )
            
            self.f = af.select(self.q1_center_new > self.q1_end,
                               f_right,
                               self.f
                              )

    elif(self.physical_system.boundary_conditions.in_q1 == 'mirror'):

        if(i_q1_start == 0):
            self.f[:N_ghost] = af.flip(self.f[N_ghost:2 * N_ghost], 0)

        if(i_q1_end == self.N_q1 - 1):
            self.f[-N_ghost:] = af.flip(self.f[-2 * N_ghost:-N_ghost], 0)

    elif(self.physical_system.boundary_conditions.in_q1 == 'shear'):
        # Not-implemented
        pass
        
    elif(self.physical_system.boundary_conditions.in_q1 == 'periodic'):
        pass

    else:
        raise NotImplementedError('Boundary condition invalid/not-implemented')

    if(self.physical_system.boundary_conditions.in_q2 == 'dirichlet'):

        if(i_q2_start == 0):
            f_bot = self.physical_system.boundary_conditions.\
                    f_bot(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )

            self.f = af.select(self.q2_center_new < self.q2_start,
                               f_bot,
                               self.f
                              )

        if(i_q2_end == self.N_q2 - 1):
            
            f_top = self.physical_system.boundary_conditions.\
                    f_top(self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.physical_system.params
                         )
            
            self.f = af.select(self.q2_center_new > self.q2_end,
                               f_top,
                               self.f
                              )

    elif(self.physical_system.boundary_conditions.in_q2 == 'mirror'):

        if(i_q2_start == 0):
            self.f[:, :N_ghost]  = af.flip(self.f[:, N_ghost:2 * N_ghost], 1)

        if(i_q2_end == self.N_q2 - 1):
            self.f[:, -N_ghost:] = af.flip(self.f[:, -2 * N_ghost:-N_ghost], 1)

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

    self._communicate_distribution_function()

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
