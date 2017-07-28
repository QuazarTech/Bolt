import numpy as np
import arrayfire as af

def f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params):
  
  m = params.mass_particle
  k = params.boltzmann_constant

  if(params.p_dim == 3):
    f0 = n * (m/(2*np.pi*k*T))**(3/2) * \
         af.exp(-m*(p1 - p1_bulk)**2/(2*k*T)) * \
         af.exp(-m*(p2 - p2_bulk)**2/(2*k*T)) * \
         af.exp(-m*(p3 - p3_bulk)**2/(2*k*T))

  elif(params.p_dim == 2):
    f0 = n * (m/(2*np.pi*k*T)) * \
         af.exp(-m*(p1 - p1_bulk)**2/(2*k*T)) * \
         af.exp(-m*(p2 - p2_bulk)**2/(2*k*T))

  else:
    f0 = n*af.sqrt(m/(2*np.pi*k*T))*\
           af.exp(-m*(p1-p1_bulk)**2/(2*k*T))

  af.eval(f0)
  return(f0)

def BGK(f, q1, q2, p1, p2, p3, moments, params):

  n = af.tile(moments('density'), 1, 1, q1.shape[2])
  T = af.tile((1/params.p_dim) * moments('energy'), 1, 1, q1.shape[2])/n
  
  p1_bulk = af.tile(moments('mom_p1_bulk'), 1, 1, q1.shape[2])/n
  p2_bulk = af.tile(moments('mom_p2_bulk'), 1, 1, q1.shape[2])/n
  p3_bulk = af.tile(moments('mom_p3_bulk'), 1, 1, q1.shape[2])/n

  C_f = -(f - \
          f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params)
         )/params.tau(q1, q2, p1, p2, p3)

  return(C_f)