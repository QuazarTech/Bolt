import numpy as np
import arrayfire as af

def f_MB(p1, p2, p3, n, T, p1_bulk, p2_bulk, p3_bulk, params):
  
  mass_particle      = params.mass_particle
  boltzmann_constant = params.boltzmann_constant

  # Assigning 'be'(backend) to the np/af module object in the case of the linear/nonlinear code:
  if(str(type(p1)) =="<class 'numpy.ndarray'>"):
    be = np 
  else:
    be = af

  if(params.p_dim == 3):
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T))**(3/2) * \
           be.exp(-mass_particle*(p1 - p1_bulk)**2/(2*boltzmann_constant*T)) * \
           be.exp(-mass_particle*(p2 - p2_bulk)**2/(2*boltzmann_constant*T)) * \
           be.exp(-mass_particle*(p3 - p3_bulk)**2/(2*boltzmann_constant*T))

  elif(params.p_dim == 2):
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           be.exp(-mass_particle*(p1 - p1_bulk)**2/(2*boltzmann_constant*T)) * \
           be.exp(-mass_particle*(p2 - p2_bulk)**2/(2*boltzmann_constant*T))

  else:
    f_MB = n*be.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             be.exp(-mass_particle*(p1-p1_bulk)**2/(2*boltzmann_constant*T))

  return(f_MB)

def BGK(system):

  n = system.compute_moments('density')
  T = (1/system.params.p_dim) * system.compute_moments('E')/n
  
  p1_bulk = system.compute_moments('p1_bulk')/n
  p2_bulk = system.compute_moments('p2_bulk')/n
  p3_bulk = system.compute_moments('p3_bulk')/n

  # Reshaping to be able to perform batched operations with p1, p2, p3
  if(str(type(system.p1)) =="<class 'numpy.ndarray'>"):
    n = n.reshape(system.N_q1, system.N_q2, 1, 1, 1)
    T = T.reshape(system.N_q1, system.N_q2, 1, 1, 1)
    
    p1_bulk = p1_bulk.reshape(system.N_q1, system.N_q2, 1, 1, 1)
    p2_bulk = p2_bulk.reshape(system.N_q1, system.N_q2, 1, 1, 1)
    p3_bulk = p3_bulk.reshape(system.N_q1, system.N_q2, 1, 1, 1)

  C_f = -(system.f - \
          f_MB(system.p1, system.p2, system.p3, n, T, p1_bulk, p2_bulk, p3_bulk, system.params)/system.normalization_constant
         )/system.params.tau

  return(C_f)