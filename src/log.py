import logging

from . import settings

_ready = False


def get(name):
    """Return a logger under the "findpro" namespace, configuring the root handler once.

    User messages and raw model output are logged at DEBUG only; set FINDPRO_DEBUG=1 to see them.
    """
    global _ready
    if not _ready:
        logging.basicConfig(
            level=logging.DEBUG if settings.DEBUG else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _ready = True
    return logging.getLogger(f"findpro.{name}")
