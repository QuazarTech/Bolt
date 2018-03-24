from params import L_x, L_y, v_max

q1_start = 0
q1_end   = L_x
N_q1     = 128

q2_start = 0
q2_end   = L_y
N_q2     = 128

p1_start = -v_max
p1_end   =  v_max
N_p1     = 18

p2_start = -v_max
p2_end   =  v_max
N_p2     = 18

p3_start = -0.5
p3_end   = 0.5
N_p3     = 1
    
N_ghost = 3
