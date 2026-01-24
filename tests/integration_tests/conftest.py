import os
import warnings
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def pytest_configure(config):
    """Configure pytest to use webdriver-manager for ChromeDriver.

    This ensures that webdriver-manager's ChromeDriver is used instead of
    any system-installed ChromeDriver, keeping it in sync with Chrome browser version.
    """
    # Get the ChromeDriver path from webdriver-manager and add it to PATH
    chromedriver_path = ChromeDriverManager().install()
    chromedriver_dir = os.path.dirname(chromedriver_path)

    # Add to PATH so selenium can find it
    current_path = os.environ.get("PATH", "")
    if chromedriver_dir not in current_path:
        os.environ["PATH"] = f"{chromedriver_dir}:{current_path}"

    # Pre-import Plotly graph objects to initialize validator cache before threading starts
    # This prevents threading-related import errors when Dash callbacks run in threads.
    # Dash's ThreadedRunner runs the server in a separate thread, and callbacks execute
    # in threads, so we need to ensure Plotly's validator cache is initialized first.
    import plotly.graph_objects as go

    # Trigger validator imports by creating commonly used objects
    go.Pie(labels=["init"], values=[1])
    go.Bar(x=[1], y=[1])
    go.Scatter(x=[1], y=[1])
    go.Figure()

    # Suppress known deprecation warnings that don't affect functionality
    # These are from third-party dependencies and will be fixed upstream
    warnings.filterwarnings(
        "ignore",
        message=".*ipykernel.comm.Comm.*",
        category=DeprecationWarning,
        module="jupyter_dash.comms",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*HTTPResponse.getheader.*",
        category=DeprecationWarning,
        module="selenium.webdriver.remote.remote_connection",
    )


def pytest_setup_options():
    """Configure Chrome options for headless testing.

    webdriver-manager automatically handles ChromeDriver installation
    and keeps it in sync with your Chrome browser version. No manual ChromeDriver installation
    is required.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    # Additional options for modern Chrome versions
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    return options
