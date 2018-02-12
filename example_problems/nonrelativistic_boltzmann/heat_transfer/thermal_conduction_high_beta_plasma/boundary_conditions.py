in_q1_left  = 'dirichlet'
in_q1_right = 'dirichlet'

in_q2_bottom = 'periodic'
in_q2_top    = 'periodic'

# Neumann Boundary Conditions for dF/dx = 0 are obtained by performing
# Taylor series expansions till second order. Then by eliminating the
# second order terms, we get the following expressions:

# |--x--|--x--|--x--|--x--|--x--|--x--|
#   -3    -2    -1     1     2     3

# F_{-1} = (4 * F_1 - F_2) / 3
# F_{-2} = (9 * F_1 - 4 * F_2) / 5
# F_{-3} = (16 * F_1 - 9 * F_2) / 7

# The similar holds true for the right side as well

# NOTE: We have only considered 3 ghost zones. However, more can be taken

def f_left(f, t, q1, q2, p1, p2, p3, params):
    n      = params.density
    T_left = params.temperature_left
    m      = params.mass
    k      = params.boltzmann_constant

    f = n * (m / (2 * np.pi * k * T_left))**(3 / 2) \
          * af.exp(-m * p1**2 / (2 * k * T_left)) \
          * af.exp(-m * p2**2 / (2 * k * T_left)) \
          * af.exp(-m * p3**2 / (2 * k * T_left))

def f_right(f, t, q1, q2, p1, p2, p3, params):
    n       = params.density
    T_right = params.temperature_right
    m       = params.mass
    k       = params.boltzmann_constant

    f = n * (m / (2 * np.pi * k * T_right))**(3 / 2) \
          * af.exp(-m * p1**2 / (2 * k * T_right)) \
          * af.exp(-m * p2**2 / (2 * k * T_right)) \
          * af.exp(-m * p3**2 / (2 * k * T_right))

def E1_left(E1, t, q1, q2, params):
    # Assuming only three ghost zones:
    E1[:, :, 0] = (16 * E1[:, :, 3] - 9 * E1[:, :, 4]) / 7
    E1[:, :, 1] = (9  * E1[:, :, 3] - 4 * E1[:, :, 4]) / 5
    E1[:, :, 2] = (4  * E1[:, :, 3] - 1 * E1[:, :, 4]) / 3

    af.eval(E1)
    return(E1)

def E2_left(E2, t, q1, q2, params):
    # Assuming only three ghost zones:
    E2[:, :, :3] = 0
    af.eval(E2)
    return(E2)

def E3_left(E3, t, q1, q2, params):
    # Assuming only three ghost zones:
    E3[:, :, 0] = (16 * E3[:, :, 3] - 9 * E3[:, :, 4]) / 7
    E3[:, :, 1] = (9  * E3[:, :, 3] - 4 * E3[:, :, 4]) / 5
    E3[:, :, 2] = (4  * E3[:, :, 3] - 1 * E3[:, :, 4]) / 3

    af.eval(E1)
    return(E1)

def B1_left(B1, t, q1, q2, params):
    # Assuming only three ghost zones:
    B1[:, :, 0] = (16 * B1[:, :, 3] - 9 * B1[:, :, 4]) / 7
    B1[:, :, 1] = (9  * B1[:, :, 3] - 4 * B1[:, :, 4]) / 5
    B1[:, :, 2] = (4  * B1[:, :, 3] - 1 * B1[:, :, 4]) / 3

    af.eval(B1)
    return(B1)

def B2_left(B2, t, q1, q2, params):
    # Assuming only three ghost zones:
    B2[:, :, :3] = 0
    af.eval(B2)
    return(B2)

def B3_left(B3, t, q1, q2, params):
    # Assuming only three ghost zones:
    B3[:, :, 0] = (16 * B3[:, :, 3] - 9 * B3[:, :, 4]) / 7
    B3[:, :, 1] = (9  * B3[:, :, 3] - 4 * B3[:, :, 4]) / 5
    B3[:, :, 2] = (4  * B3[:, :, 3] - 1 * B3[:, :, 4]) / 3

    af.eval(B3)
    return(B3)

def E1_right(E1, t, q1, q2, params):
    # Assuming only three ghost zones:
    E1[:, :, -3] = (16 * E1[:, :, -4] - 9 * E1[:, :, -5]) / 7
    E1[:, :, -2] = (9  * E1[:, :, -4] - 4 * E1[:, :, -5]) / 5
    E1[:, :, -1] = (4  * E1[:, :, -4] - 1 * E1[:, :, -5]) / 3

    af.eval(E1)
    return(E1)

def E2_right(E2, t, q1, q2, params):
    # Assuming only three ghost zones:
    E2[:, :, :3] = 0
    af.eval(E2)
    return(E2)

def E3_right(E3, t, q1, q2, params):
    # Assuming only three ghost zones:
    E3[:, :, -3] = (16 * E3[:, :, -4] - 9 * E3[:, :, -5]) / 7
    E3[:, :, -2] = (9  * E3[:, :, -4] - 4 * E3[:, :, -5]) / 5
    E3[:, :, -1] = (4  * E3[:, :, -4] - 1 * E3[:, :, -5]) / 3

    af.eval(E1)
    return(E1)

def B1_right(B1, t, q1, q2, params):
    # Assuming only three ghost zones:
    B1[:, :, -1] = (16 * B1[:, :, -4] - 9 * B1[:, :, -5]) / 7
    B1[:, :, -2] = (9  * B1[:, :, -4] - 4 * B1[:, :, -5]) / 5
    B1[:, :, -3] = (4  * B1[:, :, -4] - 1 * B1[:, :, -5]) / 3

    af.eval(B1)
    return(B1)

def B2_right(B2, t, q1, q2, params):
    # Assuming only three ghost zones:
    B2[:, :, :3] = 0
    af.eval(B2)
    return(B2)

def B3_right(B3, t, q1, q2, params):
    # Assuming only three ghost zones:
    B3[:, :, -1] = (16 * B3[:, :, -4] - 9 * B3[:, :, -5]) / 7
    B3[:, :, -2] = (9  * B3[:, :, -4] - 4 * B3[:, :, -5]) / 5
    B3[:, :, -3] = (4  * B3[:, :, -4] - 1 * B3[:, :, -5]) / 3

    af.eval(B3)
    return(B3)
