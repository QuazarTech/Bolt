from params import L_x, L_y, v_max_e, v_max_i

q1_start = 0
q1_end   = L_x
N_q1     = 1

q2_start = 0
q2_end   = L_y
N_q2     = 1024

p1_start = [-v_max_e, -v_max_i]
p1_end   = [ v_max_e,  v_max_i]
N_p1     = 32

p2_start = [-v_max_e, -v_max_i]
p2_end   = [ v_max_e,  v_max_i]
N_p2     = 32

p3_start = [-v_max_e, -v_max_i]
p3_end   = [ v_max_e,  v_max_i]
N_p3     = 32

N_ghost = 2
