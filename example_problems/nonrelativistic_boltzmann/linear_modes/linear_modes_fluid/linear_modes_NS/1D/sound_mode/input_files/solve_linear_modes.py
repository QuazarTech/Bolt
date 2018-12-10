import numpy as np 

def solve_linear_modes(params):
    n_b = params.density_background
    T_b = params.temperature_background
    v1b = params.v1_bulk_background
    tau = params.tau(0, 0, 0, 0, 0)
    k1  = params.k_q1

    linearized_system = np.array([[-1j * k1 * v1b      , -1j * k1 * n_b,  0      ],
                                  [-1j * k1 * T_b / n_b, -1j * k1 * v1b, -1j * k1],
                                  [(k1**2 * tau * v1b**2 + 6 * T_b * k1**2 * tau * v1b**2 + 3 * T_b**2 * k1**2 * tau) / n_b,
                                   -2 * (2 * k1**2 * tau * v1b**3 + 6 * T_b * k1**2 * tau * v1b + 1j * T_b * k1),
                                   -6 * k1**2 * tau * v1b**2 - 6 * T_b * k1**2 * tau - 1j * k1 * v1b 
                                  ]
                                 ]
                                )

    eigval, eigvec = np.linalg.eig(linearized_system)

    return(eigval, eigvec)
