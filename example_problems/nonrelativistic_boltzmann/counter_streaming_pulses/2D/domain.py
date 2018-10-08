from params import L_x, L_y, v_max

q1_start = 0
q1_end   = L_x
N_q1     = 48

q2_start = 0
q2_end   = L_y
N_q2     = 48

p1_start = [-v_max, -v_max]
p1_end   = [ v_max,  v_max]
N_p1     = 48

p2_start = [-v_max, -v_max]
p2_end   = [ v_max,  v_max]
N_p2     = 48

p3_start = [-0.5, -0.5]
p3_end   = [ 0.5,  0.5]
N_p3     = 1
    
N_ghost = 3
