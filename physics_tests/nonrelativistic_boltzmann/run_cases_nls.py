import arrayfire as af
import numpy as np

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import physics_tests.nonrelativistic_boltzmann.domain as domain
import physics_tests.nonrelativistic_boltzmann.boundary_conditions as boundary_conditions
import physics_tests.nonrelativistic_boltzmann.params as params
import physics_tests.nonrelativistic_boltzmann.initialize as initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

def lowpass_filter(f):
    f_hat = af.fft(f)
    dp1   = (domain.p1_end - domain.p1_start) / domain.N_p1
    k_v   = af.tile(af.to_array(np.fft.fftfreq(domain.N_p1, dp1)), 
                    1, 1, f.shape[2], f.shape[3]
                   )
    
    # Applying the filter:
    f_hat_filtered = 0.5 * (f_hat * (  af.tanh((k_v + 0.9 * af.max(k_v)) / 0.5)
                                     - af.tanh((k_v + 0.9 * af.min(k_v)) / 0.5)
                                    )
                           )

    f_hat = af.select(af.abs(k_v) < 0.8 * af.max(k_v), f_hat, f_hat_filtered)
    f = af.real(af.ifft(f_hat))
    return(f) 

N = 2**np.arange(5, 10)
def run_cases(q_dim, p_dim, charge_electron, tau):

    params.charge[0] = charge_electron
    params.tau       = tau

    # Running the setup for all resolutions:
    for i in range(N.size):
        af.device_gc()
        domain.N_q1 = int(N[i])

        if(q_dim == 2):
            domain.N_q2 = int(N[i])
            params.k_q2 = 4 * np.pi

        if(p_dim >= 2):
         
            domain.N_p2     = 32
            domain.p2_start = -10
            domain.p2_end   = 10

        if(p_dim == 3):

            domain.N_p3     = 32
            domain.p3_start = -10
            domain.p3_end   = 10


        if(charge_electron != 0):
            domain.N_p1 = int(N[i])

            if(p_dim >= 2):
                domain.N_p2 = int(N[i])

            if(p_dim == 3):
                domain.N_p3 = int(N[i])

        params.p_dim = p_dim

        # Defining the physical system to be solved:
        system = physical_system(domain,
                                 boundary_conditions,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        nls = nonlinear_solver(system)

        # Timestep as set by the CFL condition:
        dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                          / max(domain.p1_end, domain.p2_end, domain.p3_end)

        time_array = np.arange(dt, params.t_final + dt, dt)
        # Checking that time array doesn't cross final time:
        if(time_array[-1]>params.t_final):
            time_array = np.delete(time_array, -1)

        for time_index, t0 in enumerate(time_array):
            nls.strang_timestep(dt)
            if(time_index % 25 == 0):
                nls.f = lowpass_filter(nls.f)

        nls.dump_distribution_function('dump_files/nlsf_' + str(N[i]))
