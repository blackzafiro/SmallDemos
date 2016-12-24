"""
Generic useful functions.
"""


def print_msg(*msg):
    """ Prints message to stdout with color. """
    colour_format = '0;36'
    print('\x1b[%sm%s\x1b[0m' % (colour_format, " ".join([m if isinstance(m, str) else str(m) for m in msg])))
