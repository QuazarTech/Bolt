def strang(self, op1, op2, dt):
    """
    Performs strang splitting for any 2 operators.
    This scheme is 2nd order accurate in time

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    """
    op1(self, 0.5 * dt)
    op2(self, dt)
    op1(self, 0.5 * dt)

    return    

def lie(self, op1, op2, dt):
    """
    Performs lie splitting for any 2 operators.
    This scheme is 1st order accurate in time

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    """
    op1(self, dt)
    op2(self, dt)

    return  

def swss(self, op1, op2, dt):
    """
    Performs SWSS splitting for any 2 operators.
    This scheme is 2nd order accurate in time

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    
    """
    # Storing start values:
    f_start = self.f

    E1_start = self.E1
    E2_start = self.E2
    E3_start = self.E3
    
    B1_start = self.B1
    B2_start = self.B2
    B3_start = self.B3

    # Performing e^At e^Bt
    op1(self, dt)
    op2(self, dt)

    # Storing values obtained in this order:
    f_intermediate = self.f

    E1_intermediate = self.E1
    E2_intermediate = self.E2
    E3_intermediate = self.E3
    
    B1_intermediate = self.B1
    B2_intermediate = self.B2
    B3_intermediate = self.B3

    # Reassiging starting values:
    self.f = f_start    

    self.E1 = E1_start
    self.E2 = E2_start
    self.E3 = E3_start
    
    self.B1 = B1_start
    self.B2 = B2_start
    self.B3 = B3_start

    # Performing e^Bt e^At:
    op2(self, dt)
    op1(self, dt)

    # Averaging solution:
    self.f = 0.5 * (self.f + f_intermediate)
    
    self.E1 = 0.5 * (self.E1 + E1_intermediate)
    self.E2 = 0.5 * (self.E2 + E2_intermediate)
    self.E3 = 0.5 * (self.E3 + E3_intermediate)
    
    self.B1 = 0.5 * (self.B1 + B1_intermediate)
    self.B2 = 0.5 * (self.B2 + B2_intermediate)
    self.B3 = 0.5 * (self.B3 + B3_intermediate)

    return

def jia(self, op1, op2, dt):    
    """
    Performs the splitting proposed in Jia et al(2011)
    for any 2 operators. This scheme is 4th order accurate
    for commutative operations, and 3rd order accurate for
    non-commutative operations

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    
    """
    # Storing start values:
    f_start = self.f

    E1_start = self.E1
    E2_start = self.E2
    E3_start = self.E3
    
    B1_start = self.B1
    B2_start = self.B2
    B3_start = self.B3

    _strang_split_operations(self, op1, op2, dt)

    # Storing values obtained in this order:
    f_intermediate1 = self.f

    E1_intermediate1 = self.E1
    E2_intermediate1 = self.E2
    E3_intermediate1 = self.E3
    
    B1_intermediate1 = self.B1
    B2_intermediate1 = self.B2
    B3_intermediate1 = self.B3

    # Reassiging starting values:
    self.f = f_start    

    self.E1 = E1_start
    self.E2 = E2_start
    self.E3 = E3_start
    
    self.B1 = B1_start
    self.B2 = B2_start
    self.B3 = B3_start

    _strang_split_operations(self, op2, op1, dt)
    
    # Storing values obtained in this order:
    f_intermediate2 = self.f

    E1_intermediate2 = self.E1
    E2_intermediate2 = self.E2
    E3_intermediate2 = self.E3
    
    B1_intermediate2 = self.B1
    B2_intermediate2 = self.B2
    B3_intermediate2 = self.B3

    # Reassiging starting values:
    self.f = f_start    

    self.E1 = E1_start
    self.E2 = E2_start
    self.E3 = E3_start
    
    self.B1 = B1_start
    self.B2 = B2_start
    self.B3 = B3_start
    
    _swss_split_operations(self, op1, op2, dt)
    
    self.f = (2/3)*(f_intermediate1+f_intermediate2)-(1/3)*self.f
    
    self.E1 = (2/3)*(E1_intermediate1+E1_intermediate2)-(1/3)*self.E1
    self.E2 = (2/3)*(E2_intermediate1+E2_intermediate2)-(1/3)*self.E2
    self.E3 = (2/3)*(E3_intermediate1+E3_intermediate2)-(1/3)*self.E3
    
    self.B1 = (2/3)*(B1_intermediate1+B1_intermediate2)-(1/3)*self.B1
    self.B2 = (2/3)*(B2_intermediate1+B2_intermediate2)-(1/3)*self.B2
    self.B3 = (2/3)*(B3_intermediate1+B3_intermediate2)-(1/3)*self.B3

    return
