from params import L_x, L_y, delta_v

q1_start = 0
q1_end   = L_x
N_q1     = 32

q2_start = 0
q2_end   = L_y
N_q2     = 3

N_p1     = 1
p1_start = [-0.5]
p1_end   = [ 0.5]

N_p2     = 128
p2_start = [-N_p2 * delta_v / 2]
p2_end   = [ N_p2 * delta_v / 2]

N_p3     = 128
p3_start = [-N_p3 * delta_v / 2]
p3_end   = [ N_p3 * delta_v / 2]

N_ghost = 3
