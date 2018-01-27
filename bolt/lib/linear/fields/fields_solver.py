import arrayfire as af
from bolt.lib.linear.utils.fft_funcs import fft2
import numpy as np

from .electrostatic_solver import compute_electrostatic_fields

class fields_solver(object):
    
    def __init__(self, q1, q2, k_q1, k_q2, params, 
                 rho_initial, initialize_E = None, initialize_B = None
                ):

        self.q1     = q1
        self.q2     = q2
        self.k_q1   = k_q1
        self.k_q2   = k_q2
        self.params = params

        self.N_q1 = self.q1.shape[2]
        self.N_q2 = self.q2.shape[3]

        self.initialize_E = initialize_E
        self.initialize_B = initialize_B

        self._initialize(rho_initial)

    def _initialize(self, rho_initial):

        # If option is given as user-defined:
        if(self.params.fields_initialize == 'user-defined'):
            
            E1, E2, E3 = self.initialize_E(self.q1, self.q2, self.params)
            B1, B2, B3 = self.initialize_B(self.q1, self.q2, self.params)

            # Scaling Appropriately
            self.E1_hat = 2 * fft2(E1) / (self.N_q1 * self.N_q2)
            self.E2_hat = 2 * fft2(E2) / (self.N_q1 * self.N_q2)
            self.E3_hat = 2 * fft2(E3) / (self.N_q1 * self.N_q2)
            self.B1_hat = 2 * fft2(B1) / (self.N_q1 * self.N_q2)
            self.B2_hat = 2 * fft2(B2) / (self.N_q1 * self.N_q2)
            self.B3_hat = 2 * fft2(B3) / (self.N_q1 * self.N_q2)

        # Initializing EM fields using Poisson Equation:
        else:
            compute_electrostatic_fields(self, rho_initial)

        self.fields_hat = af.join(0, af.join(0, self.E1_hat, self.E2_hat, self.E3_hat), 
                                  self.B1_hat, self.B2_hat, self.B3_hat
                                 )

        af.eval(self.fields_hat)
        return

    def update_user_defined_fields(self, time_elapsed):
        """
        Updates the EM fields array using the value that is 
        returned by the user defined function at that particular time.

        Parameters
        ----------

        time_elapsed : double
                       Time at which the field values are to be evaluated.
        """

        E1, E2, E3 = self.params.user_defined_E(self.q1,
                                                self.q2,
                                                time_elapsed
                                               )

        B1, B2, B3 = self.params.user_defined_B(self.q1,
                                                self.q2,
                                                time_elapsed
                                               )

        self.E1_hat = 2 * fft2(E1) / (self.N_q1 * self.N_q2)
        self.E2_hat = 2 * fft2(E2) / (self.N_q1 * self.N_q2)
        self.E3_hat = 2 * fft2(E3) / (self.N_q1 * self.N_q2)

        self.B1_hat = 2 * fft2(B1) / (self.N_q1 * self.N_q2)
        self.B2_hat = 2 * fft2(B2) / (self.N_q1 * self.N_q2)
        self.B3_hat = 2 * fft2(B3) / (self.N_q1 * self.N_q2)
    
        af.eval(self.E1_hat, self.E2_hat, self.E3_hat,
                self.B1_hat, self.B2_hat, self.B3_hat
               )
        return
        
    def get_fields(self):
        
        if(self.params.fields_type == 'electrodynamic'):
            
            self.E1_hat = self.fields_hat[0]
            self.E2_hat = self.fields_hat[1]
            self.E3_hat = self.fields_hat[2]
            
            self.B1_hat = self.fields_hat[3]
            self.B2_hat = self.fields_hat[4]
            self.B3_hat = self.fields_hat[5]

        return(self.E1_hat, self.E2_hat, self.E3_hat, 
               self.B1_hat, self.B2_hat, self.B3_hat
              )

    # Adding solver methods:
    compute_electrostatic_fields = compute_electrostatic_fields
