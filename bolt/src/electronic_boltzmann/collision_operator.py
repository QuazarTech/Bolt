"""Contains the function which returns the Source/Sink term."""

from petsc4py import PETSc
import numpy as np
import arrayfire as af
from .matrix_inverse import inverse_4x4_matrix
import domain

@af.broadcast
def f0_defect_constant_T(f, p1, p2, p3, params):

    mu = params.mu
    T  = params.T

    for n in range(params.collision_nonlinear_iters):

        E_upper = params.E_band
        k       = params.boltzmann_constant

        tmp         = ((E_upper - mu)/(k*T))
        denominator = (k*T**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        # TODO: Multiply with the integral measure dp1 * dp2
        a00 = af.sum(T  / denominator, 0)

        fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
        af.eval(fermi_dirac)

        zeroth_moment = f - fermi_dirac

        eqn_mass_conservation   = af.sum(zeroth_moment, 0)

        N_g = domain.N_ghost
        error_mass_conservation = af.max(af.abs(eqn_mass_conservation)[0, N_g:-N_g, N_g:-N_g])

        print("    rank = ", params.rank,
	      "||residual_defect|| = ", error_mass_conservation
	     )

        res      = eqn_mass_conservation
        dres_dmu = -a00

        delta_mu = -res/dres_dmu

        mu = mu + delta_mu

        af.eval(mu)

    # Solved for mu. Now store in params
    params.mu = mu

    # Print final residual
    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
    af.eval(fermi_dirac)

    zeroth_moment = f - fermi_dirac
    
    eqn_mass_conservation   = af.sum(zeroth_moment, 0)

    N_g = domain.N_ghost
    error_mass_conservation = af.max(af.abs(eqn_mass_conservation)[0, N_g:-N_g, N_g:-N_g])

    print("    rank = ", params.rank,
	  "||residual_defect|| = ", error_mass_conservation
	 )
    print("    rank = ", params.rank,
          "mu = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]),
          "T = ", af.mean(params.T[0, N_g:-N_g, N_g:-N_g])
         )
    PETSc.Sys.Print("    ------------------")

    return(fermi_dirac)


# Using af.broadcast, since p1, p2, p3 are of size (1, 1, Np1*Np2*Np3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0_defect(f, p1, p2, p3, params):

    # Initial guess
    mu  = params.mu
    T   = params.T

    for n in range(params.collision_nonlinear_iters):
        
        E_upper = params.E_band
        k       = params.boltzmann_constant

        tmp         = ((E_upper - mu)/(k*T))
        denominator = (k*T**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        # TODO: Multiply with the integral measure dp1 * dp2
        a00 = af.sum(T                      / denominator, 0)
        a01 = af.sum((E_upper - mu)         / denominator, 0)
        a10 = af.sum(E_upper*T              / denominator, 0)
        a11 = af.sum(E_upper*(E_upper - mu) / denominator, 0)

        # Solve Ax = b
        # where A == Jacobian,
        #       x == delta guess (correction to guess), 
        #       b = -residual

        fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
        af.eval(fermi_dirac)

        zeroth_moment =          f - fermi_dirac
        second_moment = E_upper*(f - fermi_dirac)
    
        eqn_mass_conservation   = af.sum(zeroth_moment, 0)
        eqn_energy_conservation = af.sum(second_moment, 0)

        N_g = domain.N_ghost
        error_mass_conservation   = af.max(af.abs(eqn_mass_conservation)[0, N_g:-N_g, N_g:-N_g])
        error_energy_conservation = af.max(af.abs(eqn_energy_conservation)[0, N_g:-N_g, N_g:-N_g])

        residual   = [eqn_mass_conservation, eqn_energy_conservation]
        error_norm = np.max([af.max(af.abs(residual[0])), 
                             af.max(af.abs(residual[1]))]
                           )
        print("    ||residual_defect|| = ", error_norm)

#        if (error_norm < 1e-9):
#            params.mu = mu
#            params.T  = T
#            return(fermi_dirac)

        b0  = eqn_mass_conservation
        b1  = eqn_energy_conservation

        det      =   a01*a10 - a00*a11
        delta_mu = -(a11*b0 - a01*b1)/det
        delta_T  =  (a10*b0 - a00*b1)/det

        mu = mu + delta_mu
        T  = T  + delta_T

        af.eval(mu, T)

    # Solved for (mu, T). Now store in params
    params.mu = mu
    params.T  = T

    # Print final residual
    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
    af.eval(fermi_dirac)

    zeroth_moment =          f - fermi_dirac
    second_moment = E_upper*(f - fermi_dirac)
   
    eqn_mass_conservation   = af.sum(zeroth_moment, 0)
    eqn_energy_conservation = af.sum(second_moment, 0)

    residual   = [eqn_mass_conservation, eqn_energy_conservation]
    error_norm = np.max([af.max(af.abs(residual[0])), 
                         af.max(af.abs(residual[1]))]
                       )
    print("    ||residual_defect|| = ", error_norm)
    print("    mu = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]),
           "T = ", af.mean(params.T[0, N_g:-N_g, N_g:-N_g])
         )
    print("    ------------------")

    return(fermi_dirac)

@af.broadcast
def f0_ee(f, p1, p2, p3, params):

    # Initial guess
    mu_ee       = params.mu_ee
    T_ee        = params.T_ee
    vel_drift_x = params.vel_drift_x
    vel_drift_y = params.vel_drift_y

    for n in range(params.collision_nonlinear_iters):

        E_upper = params.E_band
        k       = params.boltzmann_constant

        tmp1        = (E_upper - mu_ee - p1*vel_drift_x - p2*vel_drift_y)
        tmp         = (tmp1/(k*T_ee))
        denominator = (k*T_ee**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        a_0 = T_ee      / denominator
        a_1 = tmp1      / denominator
        a_2 = T_ee * p1 / denominator
        a_3 = T_ee * p2 / denominator

        af.eval(a_0, a_1, a_2, a_3)

        # TODO: Multiply with the integral measure dp1 * dp2
        a_00 = af.sum(a_0, 0)
        a_01 = af.sum(a_1, 0)
        a_02 = af.sum(a_2, 0)
        a_03 = af.sum(a_3, 0)

        a_10 = af.sum(E_upper * a_0, 0)
        a_11 = af.sum(E_upper * a_1, 0)
        a_12 = af.sum(E_upper * a_2, 0)
        a_13 = af.sum(E_upper * a_3, 0)

        a_20 = af.sum(p1 * a_0, 0)
        a_21 = af.sum(p1 * a_1, 0)
        a_22 = af.sum(p1 * a_2, 0)
        a_23 = af.sum(p1 * a_3, 0)

        a_30 = af.sum(p2 * a_0, 0)
        a_31 = af.sum(p2 * a_1, 0)
        a_32 = af.sum(p2 * a_2, 0)
        a_33 = af.sum(p2 * a_3, 0)

        A = [ [a_00, a_01, a_02, a_03], \
              [a_10, a_11, a_12, a_13], \
              [a_20, a_21, a_22, a_23], \
              [a_30, a_31, a_32, a_33]  \
            ]
        
        fermi_dirac = 1./(af.exp( (  E_upper - mu_ee
                                   - vel_drift_x*p1 - vel_drift_y*p2 
                                  )/(k*T_ee) 
                                ) + 1.
                         )
        af.eval(fermi_dirac)

        zeroth_moment  =         (f - fermi_dirac)
        second_moment  = E_upper*(f - fermi_dirac)
        first_moment_x =      p1*(f - fermi_dirac)
        first_moment_y =      p2*(f - fermi_dirac)

        eqn_mass_conservation   = af.sum(zeroth_moment,  0)
        eqn_energy_conservation = af.sum(second_moment,  0)
        eqn_mom_x_conservation  = af.sum(first_moment_x, 0)
        eqn_mom_y_conservation  = af.sum(first_moment_y, 0)

        residual = [eqn_mass_conservation, \
                    eqn_energy_conservation, \
                    eqn_mom_x_conservation, \
                    eqn_mom_y_conservation]

        error_norm = np.max([af.max(af.abs(residual[0])),
                             af.max(af.abs(residual[1])),
                             af.max(af.abs(residual[2])),
                             af.max(af.abs(residual[3]))
                            ]
                           )
        print("    rank = ", params.rank,
	      "||residual_ee|| = ", error_norm
	     )

#        if (error_norm < 1e-13):
#            params.mu_ee       = mu_ee      
#            params.T_ee        = T_ee       
#            params.vel_drift_x = vel_drift_x
#            params.vel_drift_y = vel_drift_y
#            return(fermi_dirac)

        b_0 = eqn_mass_conservation  
        b_1 = eqn_energy_conservation
        b_2 = eqn_mom_x_conservation 
        b_3 = eqn_mom_y_conservation 
        b   = [b_0, b_1, b_2, b_3]

        # Solve Ax = b
        # where A == Jacobian,
        #       x == delta guess (correction to guess), 
        #       b = -residual

        A_inv = inverse_4x4_matrix(A)

        x_0 = A_inv[0][0]*b[0] + A_inv[0][1]*b[1] + A_inv[0][2]*b[2] + A_inv[0][3]*b[3]
        x_1 = A_inv[1][0]*b[0] + A_inv[1][1]*b[1] + A_inv[1][2]*b[2] + A_inv[1][3]*b[3]
        x_2 = A_inv[2][0]*b[0] + A_inv[2][1]*b[1] + A_inv[2][2]*b[2] + A_inv[2][3]*b[3]
        x_3 = A_inv[3][0]*b[0] + A_inv[3][1]*b[1] + A_inv[3][2]*b[2] + A_inv[3][3]*b[3]

        delta_mu = x_0
        delta_T  = x_1
        delta_vx = x_2
        delta_vy = x_3
    
        mu_ee       = mu_ee       + delta_mu
        T_ee        = T_ee        + delta_T
        vel_drift_x = vel_drift_x + delta_vx
        vel_drift_y = vel_drift_y + delta_vy

        af.eval(mu_ee, T_ee, vel_drift_x, vel_drift_y)

    # Solved for (mu_ee, T_ee, vel_drift_x, vel_drift_y). Now store in params
    params.mu_ee       = mu_ee      
    params.T_ee        = T_ee       
    params.vel_drift_x = vel_drift_x
    params.vel_drift_y = vel_drift_y

    fermi_dirac = 1./(af.exp( (  E_upper - mu_ee
                               - vel_drift_x*p1 - vel_drift_y*p2 
                              )/(k*T_ee) 
                            ) + 1.
                     )
    af.eval(fermi_dirac)

    zeroth_moment  =          f - fermi_dirac
    second_moment  = E_upper*(f - fermi_dirac)
    first_moment_x =      p1*(f - fermi_dirac)
    first_moment_y =      p2*(f - fermi_dirac)
    
    eqn_mass_conservation   = af.sum(zeroth_moment,  0)
    eqn_energy_conservation = af.sum(second_moment,  0)
    eqn_mom_x_conservation  = af.sum(first_moment_x, 0)
    eqn_mom_y_conservation  = af.sum(first_moment_y, 0)

    residual = [eqn_mass_conservation, \
                eqn_energy_conservation, \
                eqn_mom_x_conservation, \
                eqn_mom_y_conservation
               ]

    error_norm = np.max([af.max(af.abs(residual[0])),
                         af.max(af.abs(residual[1])),
                         af.max(af.abs(residual[2])),
                         af.max(af.abs(residual[3]))
                        ]
                       )
    print("    rank = ", params.rank,
	  "||residual_ee|| = ", error_norm
	 )
    N_g = domain.N_ghost
    print("    rank = ", params.rank,
          "mu_ee = ", af.mean(params.mu_ee[0, N_g:-N_g, N_g:-N_g]),
          "T_ee = ", af.mean(params.T_ee[0, N_g:-N_g, N_g:-N_g]),
          "<v_x> = ", af.mean(params.vel_drift_x[0, N_g:-N_g, N_g:-N_g]),
          "<v_y> = ", af.mean(params.vel_drift_y[0, N_g:-N_g, N_g:-N_g])
         )
    PETSc.Sys.Print("    ------------------")

    return(fermi_dirac)

def RTA(f, q1, q2, p1, p2, p3, moments, params, flag = False):
    """Return BGK operator -(f-f0)/tau."""

    if(af.any_true(params.tau_defect(q1, q2, p1, p2, p3) == 0)):
        if (flag == False):
            return(0.*f)

        f = f0_defect_constant_T(f, p1, p2, p3, params)
    
        return(f)

    C_f = -(  f - f0_defect_constant_T(f, p1, p2, p3, params) \
           ) / params.tau_defect(q1, q2, p1, p2, p3) \
          -(  f - f0_ee(f, p1, p2, p3, params)
           ) / params.tau_ee(q1, q2, p1, p2, p3)
    # When (f - f0) is NaN. Dividing by np.inf doesn't give 0
    # TODO: WORKAROUND

    af.eval(C_f)
    return(C_f)

