"""Here we define the advection terms for the Boltzmann equation."""


def A_q(p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    return (p1, p2)


class _additional_terms(object):
    def __init__(self):
        return

    def T1(self, q1, q2):
        return (q1 * q2)


def A_p(q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params,
        additional_terms=None):
    """Return the terms A_p1, A_p2 and A_p3."""
    # Additional force terms can be passed through additional_terms:
    F1 = (params.charge_electron / params.mass_particle) * (E1 +
            p2 * B3 - p3 * B2)
    F2 = (params.charge_electron / params.mass_particle) * (E2 -
            p1 * B3 + p3 * B1)
    F3 = (params.charge_electron / params.mass_particle) * (E3 -
            p2 * B1 + p1 * B2)

    return (F1, F2, F3)
