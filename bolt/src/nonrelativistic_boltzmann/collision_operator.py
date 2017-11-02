"""Contains the function which returns the Source/Sink term."""

import numpy as np
import arrayfire as af

# Using af.broadcast, since p1, p2, p3 are of size (1, 1, Np1*Np2*Np3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params):
    """Return the Local MB distribution."""
    m = params.mass_particle
    k = params.boltzmann_constant


    if (params.p_dim == 3):
        f0 = n * (m / (2 * np.pi * k * T))**(3 / 2)  \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):
        f0 = n * (m / (2 * np.pi * k * T)) \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T))

    else:
        f0 = n * af.sqrt(m / (2 * np.pi * k * T)) \
               * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T))

    af.eval(f0)
    return (f0)


def BGK(f, q1, q2, p1, p2, p3, moments, params):
    """Return BGK operator -(f-f0)/tau."""
    n = moments('density')

    # Floor used to avoid 0/0 limit:
    eps = 1e-15

    p1_bulk = moments('mom_p1_bulk') / (n + eps)
    p2_bulk = moments('mom_p2_bulk') / (n + eps)
    p3_bulk = moments('mom_p3_bulk') / (n + eps)

    T =   (1 / params.p_dim) \
        * (  moments('energy') 
           - n * p1_bulk**2
           - n * p2_bulk**2
           - n * p3_bulk**2
          ) / (n + eps) + eps

    C_f = -(  f
            - f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params)
           ) / params.tau(q1, q2, p1, p2, p3)

    # When (f - f0) is NaN. Dividing by np.inf doesn't give 0
    # WORKAROUND:
    if(isinstance(params.tau(q1, q2, p1, p2, p3), af.Array) is True):
        C_f = af.select(params.tau(q1, q2, p1, p2, p3) == np.inf, 0, C_f)
        af.eval(C_f)
    
    else:
        if(params.tau(q1, q2, p1, p2, p3) == np.inf):
            C_f = 0

    return(C_f)

def linearized_BGK(delta_f_hat, p1, p2, p3, moments, params):
    """
    Returns the array that contains the values of the linearized BGK collision operator.
    The expression that has been used may be understood more clearly by referring to the
    Sage worksheet on https://goo.gl/dXarsP
    """

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = params.rho_background
    T   = params.temperature_background
  
    tau = params.tau(0, 0, p1, p2, p3)

    dp1 = p1[1, 0, 0] - p1[0, 0, 0]
    dp2 = 1
    dp3 = 1

    if(p2.shape == 1):
        dp2 = p2[0, 1, 0] - p2[0, 0, 0]
    
    if(p3.shape == 1):
        dp3 = p3[0, 0, 1] - p3[0, 0, 0]

    # Obtaining the normalization constant:
    delta_rho_hat = moments('density')
    
    delta_p1_hat = moments('mom_p1_bulk')/rho
    delta_p2_hat = moments('mom_p2_bulk')/rho
    delta_p3_hat = moments('mom_p3_bulk')/rho
    
    delta_T_hat =   (  (1 / params.p_dim) \
                     * moments('energy') 
                     - np.sum(delta_f_hat) * dp1 * dp2 * dp3
                    ) / rho

    if(params.p_dim == 3):

        expr_term_1 = 2 * np.sqrt(2) * T * delta_p1_hat * m**(5/2) * rho * p1
        expr_term_2 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * p1**2
        expr_term_3 = 2 * np.sqrt(2) * T * delta_p2_hat * m**2.5 * rho * p2
        expr_term_4 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * p2**2
        expr_term_5 = 2 * np.sqrt(2) * T * delta_p3_hat * m**2.5 * rho * p3
        expr_term_6 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * p3**2
        expr_term_7 = 2 * np.sqrt(2) * T**2 * delta_rho_hat * k * m**(3/2)
        expr_term_8 = (2 * np.sqrt(2) * T - 3 * np.sqrt(2) * delta_T_hat) * T * k * rho * m**(3/2)
        expr_term_9 = -2 * np.sqrt(2) * rho * k * T**2 * m**(3/2)

        C_f_hat = ((((  expr_term_1 + expr_term_2 + expr_term_3 
                      + expr_term_4 + expr_term_5 + expr_term_6 
                      + expr_term_7 + expr_term_8 + expr_term_9
                     ) * np.exp(-m/(2*k*T) * (p1**2 + p1**2 + p3**2))
                    )/(8 * np.pi**1.5 * T**3.5 * k**2.5)
                   ) - delta_f_hat
                  )/tau
  
    elif(params.p_dim == 2):

        expr_term_1 = delta_T_hat * m**2 * rho * p1**2 
        expr_term_2 = delta_T_hat * m**2 * rho * p2**2
        expr_term_3 = 2 * T**2 * delta_rho_hat * k * m
        expr_term_4 = 2 * (  delta_p1_hat * m**2 * rho * p1 
                           + delta_p2_hat * m**2 * rho * p2
                           - delta_T_hat * k * m * rho
                          ) * T
        
        C_f_hat = ((  (expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/(4*np.pi*k**2*T**3)
                    * np.exp(-m/(2*k*T) * (p1**2 + p2**2))
                   ) - delta_f_hat
                  )/tau
      
    else:
    
        expr_term_1 = np.sqrt(2 * m**3) * delta_T_hat * rho * p1**2
        expr_term_2 = 2 * np.sqrt(2 * m) * k * delta_rho_hat * T**2
        expr_term_3 = 2 * np.sqrt(2 * m**3) * rho * delta_p1_hat * p1 * T
        expr_term_4 = - np.sqrt(2 * m) * k * delta_T_hat * rho * T
    
        C_f_hat = ((  (expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4) 
                    * np.exp(-m * p1**2/(2 * k * T))/(4 * np.sqrt(np.pi * T**5 * k**3))
                   ) - delta_f_hat
                  )/tau
  
    return C_f_hat