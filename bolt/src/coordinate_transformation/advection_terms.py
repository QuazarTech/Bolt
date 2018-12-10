"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.
"""

# Conservative Advection terms in q-space:
def C_q(t, r, theta, rdot, thetadot, phidot, params):
    """Return the terms C_q1, C_q2."""
    return(rdot, thetadot)

def A_q(t, r, theta, rdot, thetadot, phidot, params):
    """Return the terms A_q1, A_q2."""
    return(rdot, thetadot)
   
def A_p(t, r, theta, rdot, thetadot, phidot,
        fields_solver,
        params
       ):

    A_p1 = r * thetadot**2
    A_p2 = - 2 * rdot * thetadot / r
    A_p3 = 0 * r * thetadot

    return (A_p1, A_p2, A_p3)

def C_p(t, r, theta, rdot, thetadot, phidot,
        fields_solver,
        params
       ):

    C_p1 = r * thetadot**2
    C_p2 = - 2 * rdot * thetadot / r
    C_p3 = 0 * r * thetadot

    return (C_p1, C_p2, C_p3)
