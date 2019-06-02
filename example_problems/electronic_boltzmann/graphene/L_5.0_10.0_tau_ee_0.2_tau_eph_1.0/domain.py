import numpy as np
import params

q1_start = 0.
q1_end   = 5.0
N_q1     = 10

q2_start = 0.
q2_end   = 10.
N_q2     = 20

# If N_p1 > 1, mirror boundary conditions require p1 to be
# symmetric about zero
# TODO : Check and fix discrepancy between this and the claim
# that p1_center = mu in polar representation
N_p1     =  16

# In the cartesian representation of momentum space,
# p1 = p_x (magnitude of momentum)
# p1_start and p1_end are set such that p1_center is 0
#p1_start = [-0.04]
#p1_end   =  [0.04]


# In the polar representation of momentum space,
# p1 = p_r (magnitude of momentum)
# p1_start and p1_end are set such that p1_center is mu
p1_start = [params.initial_mu - \
        16.*params.boltzmann_constant*params.initial_temperature]
p1_end   = [params.initial_mu + \
        16.*params.boltzmann_constant*params.initial_temperature]



# If N_p2 > 1, mirror boundary conditions require p2 to be
# symmetric about zero
N_p2     =  16

# In the cartesian representation of momentum space,
# p2 = p_y (magnitude of momentum)
# p2_start and p2_end are set such that p2_center is 0
#p2_start = [-0.04]
#p2_end   =  [0.04]

# In the polar representation of momentum space,
# p2 = p_theta (angle of momentum)
# N_p_theta MUST be even.
p2_start =  [-np.pi]
p2_end   =  [np.pi]

# If N_p3 > 1, mirror boundary conditions require p3 to be
# symmetric about zero

p3_start = [-0.5]
p3_end   =  [0.5]
N_p3     =  1

N_ghost = 2
