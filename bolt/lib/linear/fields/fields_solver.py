import arrayfire as af
from bolt.lib.linear.utils.fft_func import fft2
import numpy as np

from .electrostatic_solver import compute_electrostatic_fields

class fields_solver(object):
    
    def __init__(self, ls_object):

        self.ls = ls_object
        self._initialize()

    def _initialize(self):

        self.fields_hat = af.constant(0, 6, 1, self.N_q1, self.N_q2,
                                      dtype = af.Dtype.c64
                                     )

        # If option is given as user-defined:
        if(self.ls.physical_system.params.fields_initialize == 'user-defined'):
            
            E1, E2, E3 = \
                self.ls.physical_system.initial_conditions.initialize_E(self.ls.q1_center, 
                                                                        self.ls.q2_center,
                                                                        self.ls.physical_system.params
                                                                       )
            
            B1, B2, B3 = \
                self.ls.physical_system.initial_conditions.initialize_B(self.ls.q1_center,
                                                                        self.ls.q2_center,
                                                                        self.ls.physical_system.params
                                                                       )

            # Scaling Appropriately
            self.E1_hat = 2 * fft2(E1) / (self.N_q1 * self.N_q2)
            self.E2_hat = 2 * fft2(E2) / (self.N_q1 * self.N_q2)
            self.E3_hat = 2 * fft2(E3) / (self.N_q1 * self.N_q2)
            self.B1_hat = 2 * fft2(B1) / (self.N_q1 * self.N_q2)
            self.B2_hat = 2 * fft2(B2) / (self.N_q1 * self.N_q2)
            self.B3_hat = 2 * fft2(B3) / (self.N_q1 * self.N_q2)

        # Initializing EM fields using Poisson Equation:
        else:
            compute_electrostatic_fields(self)
            
        self.fields_hat[0] = self.E1_hat
        self.fields_hat[1] = self.E2_hat
        self.fields_hat[2] = self.E3_hat
        self.fields_hat[3] = self.B1_hat
        self.fields_hat[4] = self.B2_hat
        self.fields_hat[5] = self.B3_hat

        af.eval(self.fields_hat)

    def evolve_fields(self, dt):

        if(self.ls.params.fields_solver == 'fdtd'):
            # Taken care by integrator
            pass
        
        else:
            compute_electrostatic_fields(self)

        return
        
    def get_fields(self):

        E1_hat = self.fields_hat[0]
        E2_hat = self.fields_hat[1]
        E3_hat = self.fields_hat[2]

        B1_hat = self.fields_hat[3]
        B2_hat = self.fields_hat[4]
        B3_hat = self.fields_hat[5]

        return(E1_hat, E2_hat, E3_hat, B1_hat, B2_hat, B3_hat)