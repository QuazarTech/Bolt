import arrayfire as af

# Refractive index of the medium:
# TODO: Consider giving as an input using params.
# Discuss with Mani about the same
def n(q1, q2):
    n = af.constant(0, q1.shape[0], q1.shape[1], dtype = af.Dtype.f64)
    return(n)

@af.broadcast
def A_q(q1, q2, p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    c = 1 #Speed of light

    p1_hat = p1/af.sqrt(p1**2 + p2**2)
    p2_hat = p2/af.sqrt(p1**2 + p2**2)

    return (c/n(q1, q2) * p1_hat, c/n(q1, q2) * p2_hat)

# This can then be called inside A_p if needed:
# F1 = (params.char....)(E1 + ....) + T1(q1, q2, p1, p2, p3)

def A_p(q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """Return the terms A_p1, A_p2 and A_p3."""
    F1 =   (params.charge_electron / params.mass_particle) \
         * (E1 + p2 * B3 - p3 * B2)
    F2 =   (params.charge_electron / params.mass_particle) \
         * (E2 - p1 * B3 + p3 * B1)
    F3 =   (params.charge_electron / params.mass_particle) \
         * (E3 - p2 * B1 + p1 * B2)

    return (F1, F2, F3)
