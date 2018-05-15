from numpy import pi

q1_start = 0
q1_end   = 4
N_q1     = 64 

q2_start = -pi / 2
q2_end   =  pi / 2
N_q2     = 65 # setting to an odd number to have theta = 0 as well

p1_start = [0]
p1_end   = [2]
N_p1     = 64

p2_start = [0]
p2_end   = [2]
N_p2     = 32

p3_start = [-0.5]
p3_end   = [0.5]
N_p3     = 1

N_ghost = 1
