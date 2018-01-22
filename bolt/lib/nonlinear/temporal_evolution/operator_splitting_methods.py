"""
This file contains the operator splitting methods available 
while solving a time split set of equations. 

The functions in this module are descriptive of how the operator
splitting for any 2 operators is carried out. These are then called
in the timestep routine of the nonlinear solver. 
"""

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

    dt : double
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

    dt : double
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

    dt : double
         Time-step size to evolve the system
    
    """
    # Storing start values:
    f_start = self.f

    if(self.physical_system.params.EM_fields_enabled == True):
        cell_centered_EM_fields_start = self.fields_solver.cell_centered_EM_fields
        yee_grid_EM_fields_start      = self.fields_solver.yee_grid_EM_fields

    # Performing e^At e^Bt
    op1(self, dt)
    op2(self, dt)

    # Storing values obtained in this order:
    f_intermediate = self.f

    if(self.physical_system.params.EM_fields_enabled == True):
        cell_centered_EM_fields_intermediate = self.fields_solver.cell_centered_EM_fields
        yee_grid_EM_fields_intermediate      = self.fields_solver.yee_grid_EM_fields

    # Reassiging starting values:
    self.f = f_start    
    
    if(self.physical_system.params.EM_fields_enabled == True):
        self.fields_solver.cell_centered_EM_fields = cell_centered_EM_fields_start
        self.fields_solver.yee_grid_EM_fields      = yee_grid_EM_fields_start

    # Performing e^Bt e^At:
    op2(self, dt)
    op1(self, dt)

    # Averaging solution:
    self.f = 0.5 * (self.f + f_intermediate)
    
    if(self.physical_system.params.EM_fields_enabled == True):
        self.fields_solver.cell_centered_EM_fields = 0.5 * (  self.fields_solver.cell_centered_EM_fields 
                                                             + cell_centered_EM_fields_intermediate
                                                           )
        self.fields_solver.yee_grid_EM_fields      = 0.5 * (  self.fields_solver.yee_grid_EM_fields
                                                            + yee_grid_EM_fields_intermediate
                                                           )

    return

def jia(self, op1, op2, dt):    
    """
    Performs the splitting proposed in Jia et al(2011)
    for any 2 operators. This scheme is 4th order accurate
    for commutative operations, and 3rd order accurate for
    non-commutative operations

    reference:https://www.sciencedirect.com/science/article/pii/S089571771000436X

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

    dt : double
         Time-step size to evolve the system
    
    """
    # Storing start values:
    f_start = self.f

    if(self.physical_system.params.EM_fields_enabled == True):
        cell_centered_EM_fields_start = self.fields_solver.cell_centered_EM_fields
        yee_grid_EM_fields_start      = self.fields_solver.yee_grid_EM_fields

    strang(self, op1, op2, dt)

    # Storing values obtained in this order:
    f_intermediate1 = self.f
    
    if(self.physical_system.params.EM_fields_enabled == True):
        cell_centered_EM_fields_intermediate1 = self.fields_solver.cell_centered_EM_fields
        yee_grid_EM_fields_intermediate1      = self.fields_solver.yee_grid_EM_fields

    # Reassiging starting values:
    self.f = f_start    
    
    if(self.physical_system.params.EM_fields_enabled == True):
        self.fields_solver.cell_centered_EM_fields = cell_centered_EM_fields_start
        self.fields_solver.yee_grid_EM_fields      = yee_grid_EM_fields_start

    strang(self, op2, op1, dt)
    
    # Storing values obtained in this order:
    f_intermediate2 = self.f

    if(self.physical_system.params.EM_fields_enabled == True):
        cell_centered_EM_fields_intermediate2 = self.fields_solver.cell_centered_EM_fields
        yee_grid_EM_fields_intermediate2      = self.fields_solver.yee_grid_EM_fields

    # Reassiging starting values:
    self.f = f_start    
    
    if(self.physical_system.params.EM_fields_enabled == True):
        self.fields_solver.cell_centered_EM_fields = cell_centered_EM_fields_start
        self.fields_solver.yee_grid_EM_fields      = yee_grid_EM_fields_start
    
    swss(self, op1, op2, dt)
    
    self.f = (2 / 3)*(f_intermediate1 + f_intermediate2)- (1 / 3) * self.f
    
    if(self.physical_system.params.EM_fields_enabled == True):
        self.fields_solver.cell_centered_EM_fields = (2 / 3)*(  cell_centered_EM_fields_intermediate1
                                                              + cell_centered_EM_fields_intermediate2
                                                             ) - (1 / 3) * self.cell_centered_EM_fields
        self.fields_solver.yee_grid_EM_fields      = (2 / 3)*(  yee_grid_EM_fields_intermediate1
                                                              + yee_grid_EM_fields_intermediate2
                                                             ) - (1 / 3) * self.yee_grid_EM_fields

    return
