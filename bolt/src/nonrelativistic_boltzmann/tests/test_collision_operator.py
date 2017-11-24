import numpy as np
import arrayfire as af    

from bolt.src.nonrelativistic_boltzmann import collision_operator
from bolt.lib.linear_solver.linear_solver import linear_solver
from bolt.lib.linear_solver.compute_moments import compute_moments

moment_exponents = dict(density     = [0, 0, 0],
                        mom_p1_bulk = [1, 0, 0],
                        mom_p2_bulk = [0, 1, 0],
                        mom_p3_bulk = [0, 0, 1],
                        energy      = [2, 2, 2]
                        )

moment_coeffs = dict(density     = [1, 0, 0],
                     mom_p1_bulk = [1, 0, 0],
                     mom_p2_bulk = [0, 1, 0],
                     mom_p3_bulk = [0, 0, 1],
                     energy      = [0.5, 0.5, 0.5]
                    )

@af.broadcast
def MB_dist(q1, q2, p1, p2, p3, p_dim):

    # Calculating the perturbed density:
    rho = 1 + 0.01 * af.cos(2 * np.pi * q1)

    f = rho * (1 / (2 * np.pi))**(p_dim / 2) \
            * af.exp(-0.5 * p1**2) \
            * af.exp(-0.5 * p2) \
            * af.exp(-0.5 * p3) 

    af.eval(f)
    return (f)

@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return(1 * q1**0 * p1**0)

class test(object):
    def __init__(self):
        self.physical_system = type('obj', (object, ),
                                    {'params':
                                      type('obj', (object,), {'p_dim':1,
                                                              'mass_particle':  1,
                                                              'boltzmann_constant':  1,
                                                              'rho_background':  1,
                                                              'temperature_background':  1,
                                                              'p1_bulk_background':  0,
                                                              'p2_bulk_background':  0,
                                                              'p3_bulk_background':  0,
                                                              'tau': tau
                                                             }
                                          ),
                                      'moment_exponents': moment_exponents,
                                      'moment_coeffs':    moment_coeffs,
                                     }
                                    )
    
        self.q1_start = 0
        self.q2_start = 0

        self.q1_end = 1
        self.q2_end = 1

        self.N_q1 = 32 #np.random.randint(32, 512) 
        self.N_q2 = 3

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.p1_start = -10 
        self.p2_start = -0.5
        self.p3_start = -0.5

        self.p1_end = -self.p1_start
        self.p2_end = 0.5
        self.p3_end = 0.5

        self.N_p1 = 64 #np.random.randint(32, 48)
        self.N_p2 = 1
        self.N_p3 = 1

        self.dp1 = (self.p1_end - self.p1_start) / self.N_p1
        self.dp2 = (self.p2_end - self.p2_start) / self.N_p2
        self.dp3 = (self.p3_end - self.p3_start) / self.N_p3

        self.q1_center, self.q2_center = linear_solver._calculate_q_center(self)
        self.p1, self.p2, self.p3      = linear_solver._calculate_p_center(self)
    
    compute_moments = compute_moments

def test_1V():
    
    obj = test()

    obj.single_mode_evolution = False

    f_generalized = MB_dist(obj.q1_center, obj.q2_center,
                            obj.p1, obj.p2, obj.p3, 1
                           )

    C_f_hat_generalized = 2 * af.fft2(collision_operator.BGK(f_generalized, 
                                                             obj.q1_center, obj.q2_center,
                                                             obj.p1, obj.p2, obj.p3, 
                                                             obj.compute_moments,
                                                             obj.physical_system.params
                                                            )
                                     )/(obj.N_q2 * obj.N_q1)

    # Background
    C_f_hat_generalized[0, 0, :] = 0 

    # Finding the indices of the mode excited:
    i_q1_max = np.unravel_index(af.imax(af.abs(C_f_hat_generalized))[1], 
                                (obj.N_q1, obj.N_q2, 
                                 obj.N_p1 * obj.N_p2 * obj.N_p3
                                ),order = 'F'
                               )[0]

    i_q2_max = np.unravel_index(af.imax(af.abs(C_f_hat_generalized))[1], 
                                (obj.N_q1, obj.N_q2, 
                                 obj.N_p1 * obj.N_p2 * obj.N_p3
                                ),order = 'F'
                               )[1]

    obj.p1 = np.array(af.reorder(obj.p1, 1, 2, 3, 0))
    obj.p2 = np.array(af.reorder(obj.p2, 1, 2, 3, 0))
    obj.p3 = np.array(af.reorder(obj.p3, 1, 2, 3, 0))

    delta_f_hat = 0.01 * (1 / (2 * np.pi))**(1 / 2) \
                       * np.exp(-0.5 * obj.p1**2)

    f_b = ((1 / (2 * np.pi))**(1 / 2) * np.exp(-0.5 * obj.p1**2)).reshape(1, 1, 
                                                                        obj.N_p1 
                                                                      * obj.N_p2 
                                                                      * obj.N_p3
                                                                     )

    obj.q1_center = obj.q1_center.to_ndarray().reshape(obj.N_q1, obj.N_q2, 1)
    obj.q2_center = obj.q2_center.to_ndarray().reshape(obj.N_q1, obj.N_q2, 1)

    df = (  delta_f_hat.reshape(1, 1, obj.N_p1 * obj.N_p2 * obj.N_p3) \
          * np.exp(1j * (2 * np.pi * obj.q1_center))
         ).real

    f_single_mode = af.to_array(f_b + df)

    print(af.mean(af.abs(f_generalized - f_single_mode)))
    
    obj.single_mode_evolution = True

    C_f_hat_single_mode = collision_operator.linearized_BGK(delta_f_hat, obj.p1, obj.p2, obj.p3, 
                                                            obj.compute_moments,
                                                            obj.physical_system.params
                                                           )



    print(af.mean(af.abs(  af.flat(C_f_hat_generalized[i_q1_max, i_q2_max]) 
                         - af.to_array(C_f_hat_single_mode.flatten())
                        )
                 )
         )

test_1V()

