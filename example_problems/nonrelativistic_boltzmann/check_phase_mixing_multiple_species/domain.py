from params import L_x, L_y, v_max_1, v_max_2, v_max_3

q1_start = 0
q1_end   = L_x
N_q1     = 128

q2_start = 0
q2_end   = L_y
N_q2     = 3

p1_start = [-v_max_1, -v_max_2, -v_max_3]
p1_end   = [ v_max_1,  v_max_2,  v_max_3]
N_p1     = 128

p2_start = [-0.5, -0.5, -0.5]
p2_end   = [ 0.5,  0.5,  0.5]
N_p2     = 1

p3_start = [-0.5, -0.5, -0.5]
p3_end   = [ 0.5,  0.5,  0.5]
N_p3     = 1
    
N_ghost = 3
