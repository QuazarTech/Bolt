from numpy import pi
from params import k_q1

q1_start = 0
q1_end   = 2 * pi / k_q1
N_q1     = 128

q2_start = 0
q2_end   = 1
N_q2     = 3

p1_start = [-5]
p1_end   = [5]
N_p1     = 256

p2_start = [-0.5]
p2_end   = [0.5]
N_p2     = 1

p3_start = [-0.5]
p3_end   = [0.5]
N_p3     = 1

N_ghost = 3
