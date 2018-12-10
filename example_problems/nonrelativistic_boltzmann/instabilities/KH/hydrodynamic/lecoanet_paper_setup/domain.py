from params import l0, v0

q1_start = 0 * l0
q1_end   = 1 * l0
N_q1     = 512

q2_start = 0 * l0
q2_end   = 2 * l0
N_q2     = 1024

p1_start = -16 * v0
p1_end   =  16 * v0
N_p1     = 32

p2_start = -16 * v0
p2_end   =  16 * v0
N_p2     = 32

p3_start = -0.5 #-8 * v0
p3_end   =  0.5 #8 * v0
N_p3     = 1

N_ghost = 3
