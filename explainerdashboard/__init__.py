__version__ = "0.5.6"

import logging

logging.getLogger("explainerdashboard").addHandler(logging.NullHandler())

from .explainers import ClassifierExplainer, RegressionExplainer  # noqa
from .dashboards import ExplainerDashboard, ExplainerHub, InlineExplainer  # noqa
