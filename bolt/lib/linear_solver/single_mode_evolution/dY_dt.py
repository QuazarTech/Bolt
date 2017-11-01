import numpy as np

def dY_dt(self):

    k_q1 = self.physical_system.params.k_q1   
    k_q2 = self.physical_system.params.k_q2

    charge_electron = self.physical_system.params.charge_electron

    delta_f_hat  = self.Y[0]
    delta_E1_hat = self.Y[1]
    delta_E2_hat = self.Y[2]
    delta_E3_hat = self.Y[3]
    delta_B1_hat = self.Y[4]
    delta_B2_hat = self.Y[5]
    delta_B3_hat = self.Y[6]

    delta_p1_bulk = self.compute_moments('mom_p1_bulk')
    delta_p2_bulk = self.compute_moments('mom_p2_bulk')
    delta_p3_bulk = self.compute_moments('mom_p3_bulk')

    delta_J1_hat = charge_electron * delta_p1_bulk
    delta_J2_hat = charge_electron * delta_p2_bulk
    delta_J3_hat = charge_electron * delta_p3_bulk

    ddelta_E1_hat_dt = (delta_B3_hat * 1j * k_q2) - delta_J1_hat
    ddelta_E2_hat_dt = (- delta_B3_hat * 1j * k_q1) - delta_J2_hat
    ddelta_E3_hat_dt = (delta_B2_hat * 1j * k_q1 - delta_B1_hat * 1j * k_q1) - delta_J3_hat

    ddelta_B1_hat_dt = (- delta_E3_hat * 1j * k_q2)
    ddelta_B2_hat_dt = (delta_E3_hat * 1j * k_q1)
    ddelta_B3_hat_dt = (delta_E1_hat * 1j * k_q2 - delta_E2_hat * 1j * k_q1)

    fields_term =   charge_electron * (  delta_E1_hat \
                                       + delta_B3_hat * self.p2 \
                                       - delta_B2_hat * self.p3
                                      ) * self.dfdp1_background \
                  + charge_electron * (  delta_E2_hat \
                                       + delta_B1_hat * self.p3 \
                                       - delta_B3_hat * self.p1
                                      ) * self.dfdp2_background \
                  + charge_electron * (  delta_E3_hat \
                                       + delta_B2_hat * self.p1 \
                                       - delta_B1_hat * self.p2
                                      ) * self.dfdp3_background

    C_f = self.physical_system.source(self)

    ddelta_f_hat_dt = -1j * (k_q1 * self.p1 + k_q2 * self.p2) * delta_f_hat \
                      - fields_term + C_f 
  
    dY_dt = np.array([ddelta_f_hat_dt,
                      ddelta_E1_hat_dt, ddelta_E1_hat_dt, ddelta_E3_hat_dt,
                      ddelta_B1_hat_dt, ddelta_B3_hat_dt, ddelta_B3_hat_dt
                     ]
                    )
  
    return(dY_dt)