import numpy as np

def compute_electrostatic_fields(self):

    # Intializing for the electrostatic Case:
    delta_rho_hat = self.compute_moments('density')
    delta_phi_hat =   self.physical_system.params.charge_electron \
                    * delta_rho_hat/(  self.physical_system.params.k_q1**2 
                                     + self.physical_system.params.k_q2**2
                                    )

    self.delta_E1_hat = -delta_phi_hat * (1j * self.physical_system.params.k_q1)
    self.delta_E2_hat = -delta_phi_hat * (1j * self.physical_system.params.k_q2)
    self.delta_E3_hat = 0

    self.delta_B1_hat = 0 
    self.delta_B2_hat = 0 
    self.delta_B3_hat = 0 

    return
