"""Contains the function which returns the Source/Sink term."""

import numpy as np
import arrayfire as af

# Using af.broadcast, since v1, v2, v3 are of size (1, 1, Nv1*Nv2*Nv3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params, N_s):
    """Return the Local MB distribution."""
    m     = params.mass[N_s]
    k     = params.boltzmann_constant

    if (params.p_dim == 3):
        f0 = n * (m / (2 * np.pi * k * T))**(3 / 2)  \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):
        f0 = n * (m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T))

    else:
        f0 = n * af.sqrt(m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T))

    af.eval(f0)
    return (f0)

def BGK(f, q1, q2, v1, v2, v3, moments, params, N_s = 0, flag = False):
    """Return BGK operator -(f-f0)/tau."""
    n = moments('density', N_s, f)

    # Floor used to avoid 0/0 limit:
    eps = 1e-30

    v1_bulk = moments('mom_v1_bulk', N_s, f) / (n + eps)
    v2_bulk = moments('mom_v2_bulk', N_s, f) / (n + eps)
    v3_bulk = moments('mom_v3_bulk', N_s, f) / (n + eps)

    T = (1 / params.p_dim) * (  2 * moments('energy', N_s, f) 
                              - n * v1_bulk**2
                              - n * v2_bulk**2
                              - n * v3_bulk**2
                             ) / (n + eps) + eps

    if(af.any_true(params.tau(q1, q2, v1, v2, v3) == 0)):

        f_MB = f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params, N_s)
      
        if(flag == False):
            f_MB[:] = 0        

        return(f_MB)
            
    else:

        C_f = -(  f
                - f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params, N_s)
               ) / params.tau(q1, q2, v1, v2, v3)

        # When (f - f0) is NaN. Dividing by np.inf doesn't give 0
        # Setting when tau is zero we assign f = f0 manually
        # WORKAROUND:
        if(isinstance(params.tau(q1, q2, v1, v2, v3), af.Array) is True):
            C_f = af.select(params.tau(q1, q2, v1, v2, v3) == np.inf, 0, C_f)
            af.eval(C_f)
        
        else:
            if(params.tau(q1, q2, v1, v2, v3) == np.inf):
                C_f = 0

        return(C_f)

def linearized_BGK(delta_f_hat, v1, v2, v3, moments, params):
    """
    Returns the array that contains the values of the linearized BGK collision operator.
    The expression that has been used may be understood more clearly by referring to the
    Sage worksheet on https://goo.gl/dXarsP
    """

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = params.rho_background
    T   = params.temperature_background
  
    v1_b = params.v1_bulk_background
    v2_b = params.v2_bulk_background
    v3_b = params.v3_bulk_background

    # (0, 0) are dummy values for q1, q2:
    tau = params.tau(0, 0, v1, v2, v3)

    # Obtaining the normalization constant:
    delta_rho_hat = moments('density', delta_f_hat)
    
    delta_v1_hat = (moments('mom_v1_bulk', delta_f_hat) - v1_b * delta_rho_hat)/rho
    delta_v2_hat = (moments('mom_v2_bulk', delta_f_hat) - v2_b * delta_rho_hat)/rho
    delta_v3_hat = (moments('mom_v3_bulk', delta_f_hat) - v3_b * delta_rho_hat)/rho
    
    delta_T_hat =   (  (2 / params.p_dim) \
                     * moments('energy', delta_f_hat) 
                     - delta_rho_hat * T
                     - 2 * rho * v1_b * delta_v1_hat
                     - 2 * rho * v2_b * delta_v2_hat
                     - 2 * rho * v3_b * delta_v3_hat
                    ) / rho

    # NOTE: Expressions for p_dim = 3 and p_dim = 2 are only valid when the 
    # bulk velocities of the background distribution function are zero.

    if(params.p_dim == 3):

        expr_term_1 = 2 * np.sqrt(2) * T * delta_v1_hat * m**(5/2) * rho * v1
        expr_term_2 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * v1**2
        expr_term_3 = 2 * np.sqrt(2) * T * delta_v2_hat * m**2.5 * rho * v2
        expr_term_4 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * v2**2
        expr_term_5 = 2 * np.sqrt(2) * T * delta_v3_hat * m**2.5 * rho * v3
        expr_term_6 = np.sqrt(2) * delta_T_hat * m**(5/2) * rho * v3**2
        expr_term_7 = 2 * np.sqrt(2) * T**2 * delta_rho_hat * k * m**(3/2)
        expr_term_8 = (2 * np.sqrt(2) * T - 3 * np.sqrt(2) * delta_T_hat) * T * k * rho * m**(3/2)
        expr_term_9 = -2 * np.sqrt(2) * rho * k * T**2 * m**(3/2)

        C_f_hat = ((((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4 +\
                    expr_term_5 + expr_term_6 + expr_term_7 + expr_term_8 + expr_term_9
                  )*np.exp(-m/(2*k*T) * (v1**2 + v2**2 + v3**2)))/\
                    (8 * np.pi**1.5 * T**3.5 * k**2.5)
                 ) - delta_f_hat)/tau
    
    elif(params.p_dim == 2):

        expr_term_1 = delta_T_hat * m**2 * rho * v1**2 
        expr_term_2 = delta_T_hat * m**2 * rho * v2**2
        expr_term_3 = 2 * T**2 * delta_rho_hat * k * m
        expr_term_4 = 2 * (  delta_v1_hat * m**2 * rho * v1 
                           + delta_v2_hat * m**2 * rho * v2
                           - delta_T_hat * k * m * rho
                          ) * T
        
        C_f_hat = ((  (expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/(4*np.pi*k**2*T**3)
                    * np.exp(-m/(2*k*T) * (v1**2 + v2**2))
                   ) - delta_f_hat
                  )/tau
      
    else:
        expr_term_1 = 2 * np.sqrt(2 * m**3) * rho * T * v1 * delta_v1_hat
        expr_term_2 = np.sqrt(2 * m**3) * rho * (v1**2 + v1_b**2) * delta_T_hat
        expr_term_3 = 2 * np.sqrt(2 * m) * k * T**2 * delta_rho_hat
        expr_term_4 = - np.sqrt(2 * m) * k * rho * T * delta_T_hat 
        expr_term_5 = - 2 * np.sqrt(2 * m**3) * rho * v1_b \
                      * (T * delta_v1_hat + v1 * delta_T_hat)  
    
        C_f_hat = ((  (  expr_term_1 + expr_term_2 + expr_term_3 
                       + expr_term_4 + expr_term_5 
                      ) 
                    * np.exp(-m * (v1 - v1_b)**2/(2 * k * T))/(4 * np.sqrt(np.pi * T**5 * k**3))
                   ) - delta_f_hat
                  )/tau
  
    return C_f_hat
