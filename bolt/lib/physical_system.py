#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import types
import arrayfire as af
from petsc4py import PETSc

class physical_system(object):
    """
    
    An instance of this class contains details of the physical system
    being evolved. User defines this class with the information about
    the physical system such as domain sizes, resolutions and parameters
    for the simulation. The initial conditions, the advections terms and
    the source/sink term also needs to be passed as functions by the user.
    
    """

    def __init__(self,
                 domain,
                 boundary_conditions,
                 params,
                 initial_conditions,
                 advection_term,
                 source,
                 moments
                ):
        """
        domain: Object/Input parameter file
                Contains the details of the computational domain being solved for 
                such as the dimensions and the resolution. Currently bolt is only
                capable of solving on structured rectangular grids.

        boundary_conditions: Object/Input parameter file
                             Contains details of the B.C's that need to be applied
                             along each dimension. As of the moment periodic,
                             dirichlet, shearing-box and mirror boundary conditions 
                             are supported. In case of Dirichlet boundary conditions,
                             the values at the boundaries need to be specified
                             through functions.

        params: Object/Input parameter file
                This file contains details of the parameters that are to be
                used in the initialization function, in addition to standard constants.
                Additionally, it can also store the parameters that are to be used by 
                other methods of the solver object.

        initial_conditions: File/object 
                            Contains functions which takes in the arrays as generated
                            by domain, and assigns an initial value to the distribution 
                            function/fields being evolved.

        advection_terms: File/object 
                         Contains functions advection_term.A_q1, A_q2... which are
                         declared depending upon the system that is being evolved. For
                         the FVM, the convervative advection terms C_q1, C_q2 need to
                         be defined. Simlarly the advection terms for p-space A_p1, 
                         A_p2, C_p1, C_p2 are also defined here.

        source: Function
                Returns the expression that is used on the RHS.

        moments: File/object
                 File that contains the functions defining the moments.

        """
        # Checking that domain resolution and size are 
        # of the correct data-type(only of int or float):
        attributes = [a for a in dir(domain) if not a.startswith('__')]
        
        for i in range(len(attributes)):
            if((isinstance(getattr(domain, attributes[i]), int) or
                isinstance(getattr(domain, attributes[i]), float)
               ) == 0
              ):
                raise TypeError('Expected attributes of domain \
                                 to be of type int or float'
                               )

        attributes = [a for a in dir(boundary_conditions) if not a.startswith('__')]
        
        for i in range(len(attributes)):
            if(not (isinstance(getattr(boundary_conditions, attributes[i]), str) 
               or   isinstance(getattr(boundary_conditions, attributes[i]), types.FunctionType)
               or   isinstance(getattr(boundary_conditions, attributes[i]), types.ModuleType))
              ):
                raise TypeError('Expected attributes of boundary_conditions \
                                 to be of type string or functions'
                               )

        # Checking for type of initial_conditions:
        if(isinstance(initial_conditions, types.ModuleType) is False):
            raise TypeError('Expected initial_conditions to be \
                             of type function'
                           )

        # Checking for type of source_or_sink:
        if(isinstance(source, types.FunctionType) is False):
            raise TypeError('Expected source_or_sink to be of type function')

        # Checking for the types of the methods in advection_term:
        attributes = [a for a in dir(advection_term) if not a.startswith('_')]
        for i in range(len(attributes)):
            if(not(   isinstance(getattr(advection_term, attributes[i]),types.FunctionType)
                   or isinstance(getattr(advection_term, attributes[i]),types.ModuleType)
                  )
              ):
                raise TypeError('Expected attributes of advection_term \
                                 to be of type function or module'
                               )

        attributes = [a for a in dir(moments) if not a.startswith('_')]
        for i in range(len(attributes)):
            if(not(   isinstance(getattr(moments, attributes[i]),types.FunctionType)
                   or isinstance(getattr(moments, attributes[i]),types.ModuleType)
                  )
              ):
                raise TypeError('Expected attributes of moment_defs \
                                 to be of type function or module'
                               )

        # Getting resolution and size of configuration and velocity space:
        self.N_q1, self.q1_start, self.q1_end = domain.N_q1,\
                                                domain.q1_start, domain.q1_end
        self.N_q2, self.q2_start, self.q2_end = domain.N_q2,\
                                                domain.q2_start, domain.q2_end
        self.N_p1, self.p1_start, self.p1_end = domain.N_p1,\
                                                domain.p1_start, domain.p1_end
        self.N_p2, self.p2_start, self.p2_end = domain.N_p2,\
                                                domain.p2_start, domain.p2_end
        self.N_p3, self.p3_start, self.p3_end = domain.N_p3,\
                                                domain.p3_start, domain.p3_end

        # Checking that the given input parameters are physical:
        if(self.N_q1 < 0 or self.N_q2 < 0 or
           self.N_p1 < 0 or self.N_p2 < 0 or self.N_p3 < 0 or
           domain.N_ghost < 0
          ):
            raise Exception('Grid resolution for the phase \
                             space cannot be negative'
                           )

        if(self.q1_start > self.q1_end or self.q2_start > self.q2_end or
           self.p1_start > self.p1_end or self.p2_start > self.p2_end or
           self.p3_start > self.p3_end
          ):
            raise Exception('Start point cannot be placed \
                             after the end point'
                           )

        # Evaluating step size:
        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2
        self.dp1 = (self.p1_end - self.p1_start) / self.N_p1
        self.dp2 = (self.p2_end - self.p2_start) / self.N_p2
        self.dp3 = (self.p3_end - self.p3_start) / self.N_p3

        # Getting number of ghost zones, and the 
        # boundary conditions that are utilized
        self.N_ghost             = domain.N_ghost
        self.boundary_conditions = boundary_conditions

        # Placeholder for all the modules/functions:
        # These will later be called by the methods of 
        # the solver objects:linear_solver and nonlinear_solver
        self.params             = params
        self.initial_conditions = initial_conditions

        # The following functions return the advection terms 
        # as components of a tuple:
        self.A_q = advection_term.A_q
        self.C_q = advection_term.C_q

        self.A_p = advection_term.A_p
        self.C_p = advection_term.C_p

        # Assigning the function which is used in computing the term on RHS:
        # Usually, this is taken as a relaxation type collision operator
        self.source = source

        # Assigning the moments data:
        self.moments = moments

        # Declaring the MPI communicator:
        self.mpi_communicator = PETSc.COMM_WORLD.tompi4py()

        # Finding the number of species:
        N_species = len(params.charge)

        try:
            assert(len(params.mass) == len(params.charge))
        except:
            raise Exception('Inconsistenty in number of species. Mismatch between\
                             the number of species mentioned in charge and mass inputs'
                           )

        if(params.fields_type == 'electrodynamic'):
            try:
                assert(params.fields_solver.upper() == 'FDTD')
            except:
                raise Exception('Solver specified isn\'t an electrodynamic solver')

        if(params.fields_type == 'electrostatic'):
            try:
                assert(   params.fields_solver.upper() == 'SNES'
                       or params.fields_solver.upper() == 'FFT'
                      )
            except:
                raise Exception('Solver specified isn\'t an electrostatic solver')

        # Printing code signature:
        PETSc.Sys.Print('----------------------------------------------------------------------')
        PETSc.Sys.Print("|                      ,/                                            |")
        PETSc.Sys.Print("|                    ,'/          ____        ____                   |")                   
        PETSc.Sys.Print("|                  ,' /          / __ )____  / / /_                  |")
        PETSc.Sys.Print("|                ,'  /_____,    / __  / __ \/ / __/                  |")
        PETSc.Sys.Print("|              .'____    ,'    / /_/ / /_/ / / /_                    |")
        PETSc.Sys.Print("|                   /  ,'     /_____/\____/_/\__/                    |")
        PETSc.Sys.Print("|                  / ,'                                              |")
        PETSc.Sys.Print("|                 /,'                                                |")
        PETSc.Sys.Print("|                /'                                                  |")
        PETSc.Sys.Print('|--------------------------------------------------------------------|')
        PETSc.Sys.Print('|Copyright (C) 2017-18, Research Division, Quazar Technologies, Delhi|')
        PETSc.Sys.Print('|                                                                    |')
        PETSc.Sys.Print('|  Bolt is free software; you can redistribute it and/or modify it   |')
        PETSc.Sys.Print('|  under the terms of the GNU General Public License as published    |')
        PETSc.Sys.Print('|  by the Free Software Foundation(version 3.0)                      |')
        PETSc.Sys.Print('----------------------------------------------------------------------')
        PETSc.Sys.Print('Resolution(Nq1, Nq2, Np1, Np2, Np3):', '(', domain.N_q1, ',', domain.N_q2, 
                        ',',domain.N_p1, ',', domain.N_p2, ',', domain.N_p3, ')'
                       )
        PETSc.Sys.Print('Solver Method in q-space           :', params.solver_method_in_q.upper())

        if(params.solver_method_in_q == 'FVM'):
            PETSc.Sys.Print('    Reconstruction Method          :', params.reconstruction_method_in_q.upper())
            PETSc.Sys.Print('    Riemann Solver                 :', params.riemann_solver_in_q.upper())

        PETSc.Sys.Print('Solver Method in p-space           :', params.solver_method_in_p.upper())

        if(params.solver_method_in_p == 'FVM'):
            PETSc.Sys.Print('    Reconstruction Method          :', params.reconstruction_method_in_p.upper())
            PETSc.Sys.Print('    Riemann Solver                 :', params.riemann_solver_in_p.upper())

        if(params.fields_enabled == True):
            PETSc.Sys.Print('Fields Type                        :', params.fields_type.upper())
            PETSc.Sys.Print('Fields Initialization Method       :', params.fields_initialize.upper())
            PETSc.Sys.Print('Fields Solver Method               :', params.fields_solver.upper())
        PETSc.Sys.Print('Number of Species                  :', N_species)
        # for i in range(N_species):
        #     PETSc.Sys.Print('   Charge(Species %1d)               :'%(i+1), params.charge[i])
        #     PETSc.Sys.Print('   Mass(Species %1d)                 :'%(i+1), params.mass[i])
        PETSc.Sys.Print('Number of Devices/Node             :', params.num_devices)
