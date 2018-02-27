in_q1_left  = 'dirichlet'
in_q1_right = 'mirror'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):
    n       = params.density
    v1_bulk = params.v1_bulk
    beta    = params.beta

    f =   (n / (np.pi * beta)**(3 / 2)) \
        * af.exp(-(v1 - v1_bulk)**2 / beta) \
        * af.exp(-v2**2 / beta) \
        * af.exp(-v3**2 / beta)

    return(f)
