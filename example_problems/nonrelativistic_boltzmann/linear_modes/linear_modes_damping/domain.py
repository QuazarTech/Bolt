from params import p_dim, l0, v0

q1_start = 0 * l0
q1_end   = 1 * l0
N_q1     = 32

q2_start = 0 * l0
q2_end   = 1 * l0
N_q2     = 3

p1_start = [-10 * v0]
p1_end   = [10  * v0]
N_p1     = 32

if(p_dim > 1):
    p2_start = [-10 * v0]
    p2_end   = [10  * v0]
    N_p2     = 32

else:
    p2_start = [-0.5]
    p2_end   = [0.5]
    N_p2     = 1

if(p_dim == 3):
    p3_start = [-10 * v0]
    p3_end   = [10  * v0]
    N_p3     = 32

else:
    p3_start = [-0.5]
    p3_end   = [0.5]
    N_p3     = 1
    
N_ghost = 3
