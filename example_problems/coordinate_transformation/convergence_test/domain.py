from numpy import pi

q1_start = 0.5
q1_end   = 4
N_q1     = 32 

q2_start = -pi / 2
q2_end   =  pi / 2
N_q2     = 32

p1_start = [0]
p1_end   = [2]
N_p1     = 32

p2_start = [0]
p2_end   = [1 / q1_start]
N_p2     = 32

p3_start = [-0.5]
p3_end   = [0.5]
N_p3     = 1

N_ghost = 1
