import numpy as np
import arrayfire as af

def f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params):
  
  m = params.mass_particle
  k = params.boltzmann_constant

  # Assigning 'be'(backend) to the np/af module object in the case of the linear/nonlinear code:
  if(str(type(p1)) =="<class 'numpy.ndarray'>"):
    be = np 
  else:
    be = af

  if(params.p_dim == 3):
    f0 = n * (m/(2*np.pi*k*T))**(3/2) * \
         be.exp(-m*(p1 - p1_bulk)**2/(2*k*T)) * \
         be.exp(-m*(p2 - p2_bulk)**2/(2*k*T)) * \
         be.exp(-m*(p3 - p3_bulk)**2/(2*k*T))

  elif(params.p_dim == 2):
    f0 = n * (m/(2*np.pi*k*T)) * \
         be.exp(-m*(p1 - p1_bulk)**2/(2*k*T)) * \
         be.exp(-m*(p2 - p2_bulk)**2/(2*k*T))

  else:
    f0 = n*be.sqrt(m/(2*np.pi*k*T))*\
           be.exp(-m*(p1-p1_bulk)**2/(2*k*T))

  return(f0)

def BGK(f, q1, q2, p1, p2, p3, moments, params):

  n = moments('density')
  T = (1/params.p_dim) * moments('energy')/n
  
  p1_bulk = moments('p1_bulk')/n
  p2_bulk = moments('p2_bulk')/n
  p3_bulk = moments('p3_bulk')/n

  # Reshaping to be able to perform batched operations with p1, p2, p3
  if(str(type(p1)) =="<class 'numpy.ndarray'>"):
    n = n.reshape(n.shape[0], n.shape[1], 1, 1, 1)
    T = T.reshape(T.shape[0], T.shape[1], 1, 1, 1)
    
    p1_bulk = p1_bulk.reshape(p1_bulk.shape[0], p1_bulk.shape[1], 1, 1, 1)
    p2_bulk = p2_bulk.reshape(p2_bulk.shape[0], p2_bulk.shape[1], 1, 1, 1)
    p3_bulk = p3_bulk.reshape(p3_bulk.shape[0], p3_bulk.shape[1], 1, 1, 1)

  C_f = -(f - \
          f0(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params)/params.normalization_constant
         )/0.01 #params.tau(q1, q2, p1, p2, p3)

  return(C_f)