from params import l0, v0
from numpy import pi

q1_start = -2 * pi * l0
q1_end   = 2 * pi * l0
N_q1     = 64

q2_start = 0 * l0
q2_end   = 1 * l0
N_q2     = 3

# Velocity
p1_start = [-15 * v0]
p1_end   = [15  * v0]
N_p1     = 64

p2_start = [-0.5 * v0]
p2_end   = [0.5  * v0]
N_p2     = 1

p3_start = [-0.5 * v0]
p3_end   = [0.5  * v0]
N_p3     = 1

N_ghost = 3
