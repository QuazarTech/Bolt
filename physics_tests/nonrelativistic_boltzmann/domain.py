from numpy import pi
from .params import l0, v0

q1_start = 0      * l0
q1_end   = 5 * pi * l0
N_q1     = 128

q2_start = 0 * l0
q2_end   = 1 * l0
N_q2     = 3

# Velocity
p1_start = -90 * v0
p1_end   = 90  * v0
N_p1     = 1024

p2_start = -0.5 * v0
p2_end   = 0.5  * v0
N_p2     = 1

p3_start = -0.5 * v0
p3_end   = 0.5  * v0
N_p3     = 1

N_ghost = 3
