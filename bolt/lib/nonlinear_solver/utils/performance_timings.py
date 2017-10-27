#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from prettytable import PrettyTable

def print_table(self, N_iters):
    """
    This function is used to check the timings
    of each of the functions which are used during the 
    process of a single-timestep.
    """

    # Initializing the global variables(timespent per timestep):
    time_ts = np.zeros(1) 
    time_interp2 = np.zeros(1); time_sourcets = np.zeros(1)
    time_fvm_solver = np.zeros(1); time_reconstruct = np.zeros(1)
    time_riemann = np.zeros(1); time_fvm_ts = np.zeros(1)
    time_fieldstep = np.zeros(1); time_fieldsolver = np.zeros(1); time_interp3 = np.zeros(1)
    time_communicate_f = np.zeros(1); time_communicate_fields = np.zeros(1) 
    time_apply_bcs_f = np.zeros(1); time_apply_bcs_fields = np.zeros(1)

    # Performing reduction operations to obtain the greatest time amongst nodes/devices:
    self._comm.Reduce(np.array([self.time_ts/N_iters]), time_ts,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_interp2/N_iters]), time_interp2,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_sourcets/N_iters]), time_sourcets,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_fvm_solver/N_iters]), time_fvm_solver,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_reconstruct/N_iters]), time_reconstruct,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_riemann/N_iters]), time_riemann,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_fvm_ts/N_iters]), time_fvm_ts,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_communicate_f/N_iters]), time_communicate_f,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_apply_bcs_f/N_iters]), time_apply_bcs_f,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_fieldstep/N_iters]), time_fieldstep,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_fieldsolver/N_iters]), time_fieldsolver,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_interp3/N_iters]), time_interp3,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_communicate_fields/N_iters]), time_communicate_fields,
                      op = MPI.MAX, root = 0
                     )
    self._comm.Reduce(np.array([self.time_apply_bcs_fields/N_iters]), time_apply_bcs_fields,
                      op = MPI.MAX, root = 0
                     )
                     
    if(self._comm.rank == 0):

        table = PrettyTable(["Method", "Time-Taken(s/iter)", "Percentage(%)"])
        table.add_row(['TIMESTEP', time_ts[0], 100])
        
        table.add_row(['Q_ADVECTION', time_interp2[0],
                       100*time_interp2[0]/time_ts[0]
                      ]
                     )
        
        table.add_row(['SOURCE_TS', time_sourcets[0],
                       100*time_sourcets[0]/time_ts[0]
                      ]
                     )

        table.add_row(['FVM_SOLVER', time_fvm_solver[0],
                       100 * time_fvm_solver[0]/time_ts[0]
                      ]
                     )

        table.add_row(['FIELD-STEP', time_fieldstep[0],
                       100*time_fieldstep[0]/time_ts[0]
                      ]
                     )

        table.add_row(['APPLY_BCS_F', time_apply_bcs_f[0],
                       100*time_apply_bcs_f[0]/time_ts[0]
                      ]
                     )

        table.add_row(['COMMUNICATE_F', time_communicate_f[0],
                       100*time_communicate_f[0]/time_ts[0]
                      ]
                     )
   
        PETSc.Sys.Print(table)

        if(self.physical_system.params.charge_electron != 0):

            PETSc.Sys.Print('FIELDS-STEP consists of:')
            
            table = PrettyTable(["Method", "Time-Taken(s/iter)", "Percentage(%)"])

            table.add_row(['FIELD-STEP', time_fieldstep[0],
                           100
                          ]
                         )

            table.add_row(['FIELD-SOLVER', time_fieldsolver[0],
                           100*time_fieldsolver[0]/time_fieldstep[0]
                          ]
                         )

            table.add_row(['P_ADVECTION', time_interp3[0],
                           100*time_interp3[0]/time_fieldstep[0]
                          ]
                         )

            table.add_row(['APPLY_BCS_FIELDS', time_apply_bcs_fields[0],
                           100*time_apply_bcs_fields[0]/time_fieldstep[0]
                          ]
                         )

            table.add_row(['COMMUNICATE_FIELDS', time_communicate_fields[0],
                           100*time_communicate_fields[0]/time_fieldstep[0]
                          ]
                         )

            PETSc.Sys.Print(table)

        if(self.physical_system.params.solver_method_in_q == 'FVM'):

            PETSc.Sys.Print('FVM_SOLVER consists of:')
            
            table = PrettyTable(["Method", "Time-Taken(s/iter)", "Percentage(%)"])

            table.add_row(['FVM_SOLVER', time_fvm_solver[0],
                           100
                          ]
                         )

            table.add_row(['RECONSTRUCTION', time_reconstruct[0],
                           100*time_reconstruct[0]/time_fvm_solver[0]
                          ]
                         )

            table.add_row(['RIEMANN-SOLVER', time_riemann[0],
                           100*time_riemann[0]/time_fvm_solver[0]
                          ]
                         )

            table.add_row(['DFDT_INTEGRATION', time_fvm_ts[0],
                           100*time_fvm_ts[0]/time_fvm_solver[0]
                          ]
                         )

            PETSc.Sys.Print(table)

        PETSc.Sys.Print('Spatial Zone Cycles/s =', self.N_q1 * self.N_q2 / time_ts[0])
