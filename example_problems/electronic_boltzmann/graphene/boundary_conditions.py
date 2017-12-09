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
    T       = 0.*params.T + 3e-4*4.
    
    f_left = (1./(af.exp( (E_upper - 0.*0.01)/(k*T) ) + 1.)
             )

    # Inflow between y = [y_center(j_inflow_start), y_center(j_inflow_end)]
    N_q2               = domain.N_q2
    N_g                = domain.N_ghost
    size_of_inflow     = 5.
    offset_from_center = 0.
    length_y           = domain.q2_end - domain.q2_start
    N_inflow_zones     = (int)(size_of_inflow/length_y*N_q2)
    N_offset           = (int)(abs(offset_from_center)/length_y*N_q2)
    j_inflow_start     =   N_g + N_q2/2 - N_inflow_zones/2 \
                         + np.sign(offset_from_center)*N_offset
    j_inflow_end       =   N_g + N_q2/2 + N_inflow_zones/2 \
                         + np.sign(offset_from_center)*N_offset

    f_left[:, :, :j_inflow_start] = f[:, :, :j_inflow_start]
    f_left[:, :, j_inflow_end:]   = f[:, :, j_inflow_end:]

    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = 0.*params.T + 3e-4*4.

    f_right = (1./(af.exp( (E_upper - 0.*0.01)/(k*T) ) + 1.)
              )

    # Outflow between y = [y_center(j_outflow_start), y_center(j_outflow_end)]
    N_q2               = domain.N_q2
    N_g                = domain.N_ghost
    size_of_outflow    = 5.
    offset_from_center = 0.
    length_y           = domain.q2_end - domain.q2_start
    N_outflow_zones    = (int)(size_of_outflow/length_y*N_q2)
    N_offset           = (int)(abs(offset_from_center)/length_y*N_q2)
    j_outflow_start    =   N_g + N_q2/2 - N_outflow_zones/2 \
                         + np.sign(offset_from_center)*N_offset
    j_outflow_end      =   N_g + N_q2/2 + N_outflow_zones/2 \
                         + np.sign(offset_from_center)*N_offset

    f_right[:, :, :j_outflow_start] = f[:, :, :j_outflow_start]
    f_right[:, :, j_outflow_end:]   = f[:, :, j_outflow_end:]

    af.eval(f_right)
    return(f_right)
