import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror'

@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = params.vel_drift_x_in  * np.sin(omega*t)

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p1 - mu)/(k*T) ) + 1.)
                     )

    if (params.contact_geometry="straight"):
        # Contacts on either side of the device

        q2_contact_start = params.contact_start
        q2_contact_end   = params.contact_end
        
        cond = ((q2 >= q2_contact_start) & \
                (q2 <= q2_contact_end) \
               )

        f_left = cond*fermi_dirac_in + (1 - cond)*f

    elif (params.contact_geometry="turn_around"):
        # Contacts on the same side of the device

        vel_drift_x_out = -params.vel_drift_x_in * np.sin(omega*t)

        fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p1 - mu)/(k*T) ) + 1.)
                          )
    
        # TODO: set these parameters in params.py
        cond_in  = ((q2 >= 3.5) & (q2 <= 4.5))
        cond_out = ((q2 >= 5.5) & (q2 <= 6.5))
    
        f_left =  cond_in*fermi_dirac_in + cond_out*fermi_dirac_out \
                + (1 - cond_in)*(1 - cond_out)*f

    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_out = params.vel_drift_x_out * np.sin(omega*t)

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p1 - mu)/(k*T) ) + 1.)
                      )
    
    if (params.contact_geometry="straight"):
        # Contacts on either side of the device

        q2_contact_start = params.contact_start
        q2_contact_end   = params.contact_end
        
        cond = ((q2 >= q2_contact_start) & \
                (q2 <= q2_contact_end) \
               )

        f_right = cond*fermi_dirac_out + (1 - cond)*f

    elif (params.contact_geometry="turn_around"):
        # Contacts on the same side of the device
        
        f_right = f

    af.eval(f_right)
    return(f_right)
