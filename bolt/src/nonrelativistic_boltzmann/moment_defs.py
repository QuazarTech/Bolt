def density(f, v1, v2, v3):
    return(f)

def mom_v1_bulk(f, v1, v2, v3):
    return(f * v1)

def mom_v2_bulk(f, v1, v2, v3):
    return(f * v2)

def mom_v3_bulk(f, v1, v2, v3):
    return(f * v3)

def energy(f, v1, v2, v3):
    return(f * (v1**2 + v2**2 + v3**2))

def q_q1(f, v1, v2, v3):
    return(f * v1 * (v1**2 + v2**2 + v3**2))

def q_q2(f, v1, v2, v3):
    return(f * v2 * (v1**2 + v2**2 + v3**2))

def q_q3(f, v1, v2, v3):
    return(f * v3 * (v1**2 + v2**2 + v3**2))
