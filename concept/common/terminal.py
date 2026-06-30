"""
concept/common/terminal.py
Non-blocking terminal I/O — identical across all concept simulations.

In a GUI context these are unused; they exist only for the standalone
CLI scripts that originally used a keypress loop to pause/inspect runs.
"""
import os, select, sys

try:
    import termios, tty
    _HAS_TERMIOS = True
except ImportError:          # Windows
    _HAS_TERMIOS = False


class NullIO:
    """A do-nothing output sink, used to silence a sim's `fout` in GUI mode."""
    def write(self, *a, **k):  return 0
    def flush(self):           pass

_saved_term_settings = None


def setnodelay(fd):
    """Put the terminal into raw, non-blocking-read mode."""
    global _saved_term_settings
    if not _HAS_TERMIOS:
        return
    try:
        _saved_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except termios.error:
        _saved_term_settings = None


def setnormal(fd):
    """Restore the terminal to its normal (canonical/echo) mode."""
    global _saved_term_settings
    if not _HAS_TERMIOS:
        return
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
        if _saved_term_settings is not None:
            termios.tcsetattr(fd, termios.TCSANOW, _saved_term_settings)
    except termios.error:
        pass


def read_nonblocking(fd):
    """Return one character if available without blocking, else None."""
    if not _HAS_TERMIOS:
        return None
    r, _, _ = select.select([fd], [], [], 0)
    if r:
        ch = os.read(fd, 1)
        if ch:
            return ch.decode(errors="replace")
    return None


def read_blocking_char(fd):
    """Block until a single character is available, and return it."""
    while True:
        r, _, _ = select.select([fd], [], [], None)
        if r:
            ch = os.read(fd, 1)
            if ch:
                return ch.decode(errors="replace")
