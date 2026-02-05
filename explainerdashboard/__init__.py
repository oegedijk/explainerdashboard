__version__ = "0.5.6"

import logging
import sys

logging.getLogger("explainerdashboard").addHandler(logging.NullHandler())


def enable_default_logging(level=logging.INFO):
    """Enable stdout logging for explainerdashboard."""
    logger = logging.getLogger("explainerdashboard")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


from .explainers import ClassifierExplainer, RegressionExplainer  # noqa
from .dashboards import ExplainerDashboard, ExplainerHub, InlineExplainer  # noqa
