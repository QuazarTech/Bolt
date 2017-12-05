# The following function is used in formatting for print:
def indent(txt, stops=1):
    """
    This function indents every line of the input as a multiple of
    4 spaces.
    """
    return '\n'.join(" " * 4 * stops + line for line in  txt.splitlines())
