"""Where dstoolkit's one-line summaries go ("Filled missing values: age (263)", ...).

Every message is sent to the standard `logging` logger named "dstoolkit" at
INFO level. What happens next depends on set_message_output():

- "print" (default): the logger writes the message to the screen (standard
  output), exactly like print(), so notebooks and scripts show it.
- "logging": the logger hands messages to your own logging setup instead, so
  they get your timestamps and format and go wherever your logs go (a file,
  AWS CloudWatch from a Lambda function, ...).
- "off": nothing is shown anywhere.

`verbose=False` on any function still silences that one call.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

MessageOutput = Literal["print", "logging", "off"]

logger = logging.getLogger("dstoolkit")


class _StdoutHandler(logging.Handler):
    """Writes the bare message to whatever sys.stdout is at that moment.

    Looking sys.stdout up on every message (rather than once, like
    logging.StreamHandler) keeps Jupyter, pytest's capsys and
    contextlib.redirect_stdout working.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
        except Exception:  # pragma: no cover - logging must never crash the caller
            self.handleError(record)


_console = _StdoutHandler()
_output: MessageOutput = "print"


def set_message_output(output: MessageOutput) -> None:
    """Choose where dstoolkit's summaries go: "print" (default), "logging" or "off".

    "logging" sends them to the "dstoolkit" logger at INFO level and lets your
    logging configuration handle them, for example:

        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        dst.set_message_output("logging")

    The setting applies to the whole Python session.
    """
    global _output
    if output not in ("print", "logging", "off"):
        raise ValueError(f'output must be "print", "logging" or "off", got {output!r}')
    _output = output
    # Remove every copy of the print handler, including one left by an earlier import of
    # this module (importlib.reload or Jupyter's %autoreload), so messages never print twice
    for handler in list(logger.handlers):
        if type(handler).__name__ == "_StdoutHandler":
            logger.removeHandler(handler)
    if output == "print":
        logger.addHandler(_console)
        logger.setLevel(logging.INFO)
        logger.propagate = False      # don't also pass it to the app's logging setup
    elif output == "logging":
        logger.setLevel(logging.NOTSET)   # the app's logging configuration decides
        logger.propagate = True
    else:
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False


def get_message_output() -> MessageOutput:
    """The current setting of set_message_output()."""
    return _output


def say(verbose: bool, message: str) -> None:
    """Send one summary message, unless this call was made with verbose=False."""
    if verbose:
        logger.info(message)


if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())   # standard library practice for packages
set_message_output("print")
