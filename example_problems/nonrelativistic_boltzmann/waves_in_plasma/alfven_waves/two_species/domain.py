from params import L_x, L_y, v_max_e, v_max_i 

q1_start = 0
q1_end   = L_x
N_q1     = 32

q2_start = 0
q2_end   = L_y
N_q2     = 3

p1_start = [-0.5, -0.5]
p1_end   = [ 0.5,  0.5]
N_p1     = 1

p2_start = [-v_max_e, -v_max_i]
p2_end   = [ v_max_e,  v_max_i]
N_p2     = 128

p3_start = [-v_max_e, -v_max_i]
p3_end   = [ v_max_e,  v_max_i]
N_p3     = 128
    
N_ghost = 3
