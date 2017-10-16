"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.
"""

# Conservative Advection terms in q-space:
def C_q(p1, p2, p3, params):
    """Return the terms C_q1, C_q2."""
    return (p1, p2)

def A_q(p1, p2, p3, params):
    """Return the terms A_q1, A_q2."""
    return (p1, p2)

# If necessary, additional terms as a function of the arguments
# passed to A_p may be used:s
# For instance:
def T1(q1, q2, p1, p2, p3):
    return(q1*q2)

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
