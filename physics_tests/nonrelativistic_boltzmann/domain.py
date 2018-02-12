from .params import p_dim

q1_start = 0
q1_end   = 1
N_q1     = 32

q2_start = 0
q2_end   = 1
N_q2     = 3

p1_start = -10
p1_end   = 10
N_p1     = 32

if(p_dim > 1):
    p2_start = -10
    p2_end   = 10
    N_p2     = 32

else:
    p2_start = -0.5
    p2_end   = 0.5
    N_p2     = 1

if(p_dim == 3):
    p3_start = -10
    p3_end   = 10
    N_p3     = 32

else:
    p3_start = -0.5
    p3_end   = 0.5
    N_p3     = 1
    
N_ghost = 3
