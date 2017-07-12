def A_q(p1, p2, p3, params):
  return(p1, p2)

def A_p(q1, q2, p1, p2, p3, E1, E2, E3, B1, B2, B3, *args):
  # Additional force terms can be passed through args
  return(0 * p1, 0 * p2, 0 * p3)