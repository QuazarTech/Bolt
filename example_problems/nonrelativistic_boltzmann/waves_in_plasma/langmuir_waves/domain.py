from numpy import pi
from params import l0, v0

q1_start = 0      #* l0
q1_end   = 5 * pi #* l0
N_q1     = 128

q2_start = 0      #* l0
q2_end   = 1      #* l0
N_q2     = 3

p1_start = -60    #* v0
p1_end   =  60    #* v0

dv = T0/10
N_p1     = (int)(p1_end - p1_start)/dv 

p2_start = -0.5
p2_end   =  0.5
N_p2     = 1

p3_start = -0.5
p3_end   = 0.5
N_p3     = 1

N_ghost_q = 3
N_ghost_p = 0
