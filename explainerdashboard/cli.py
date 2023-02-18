"""
Command-line tool for starting an explainerdashboard from a particular directory
"""
import os
import webbrowser
from pathlib import Path
from importlib import import_module
from copy import deepcopy
import pickle
import oyaml as yaml

import pandas as pd
import joblib
import click

import waitress

from explainerdashboard import *
from explainerdashboard.explainers import BaseExplainer
from explainerdashboard.dashboards import ExplainerDashboard


explainer_ascii = """

 _____ ___ __| |__ _(_)_ _  ___ _ _ __| |__ _ __| |_ | |__  ___  __ _ _ _ __| |
/ -_) \ / '_ \ / _` | | ' \/ -_) '_/ _` / _` (_-< ' \| '_ \/ _ \/ _` | '_/ _` |
\___/_\_\ .__/_\__,_|_|_||_\___|_| \__,_\__,_/__/_||_|_.__/\___/\__,_|_| \__,_|
        |_| 

"""

hub_ascii = """

               _      _              _        _    
  _____ ___ __| |__ _(_)_ _  ___ _ _| |_ _  _| |__ 
 / -_) \ / '_ \ / _` | | ' \/ -_) '_| ' \ || | '_ \
 \___/_\_\ .__/_\__,_|_|_||_\___|_| |_||_\_,_|_.__/
         |_|                                      

"""


def build_explainer(explainer_config):
    if isinstance(explainer_config, (Path, str)) and str(explainer_config).endswith(
        ".yaml"
    ):
        config = yaml.safe_load(open(str(explainer_config), "r"))
    elif isinstance(explainer_config, dict):
        config = explainer_config
    assert (
        "explainer" in config
    ), "Please pass a proper explainer.yaml config file that starts with `explainer:`!"
    config = explainer_config["explainer"]

    print(f"explainerdashboard ===> Loading model from {config['modelfile']}")
    model = pickle.load(open(config["modelfile"], "rb"))

    print(f"explainerdashboard ===> Loading data from {config['datafile']}")
    if str(config["datafile"]).endswith(".csv"):
        df = pd.read_csv(config["datafile"])
    elif str(config["datafile"]).endswith(".parquet"):
        df = pd.read_parquet(config["datafile"])
    else:
        raise ValueError("datafile should either be a .csv or .parquet!")

    print(
        f"explainerdashboard ===> Using column {config['data_target']} to generate X, y "
    )
    target_col = config["data_target"]
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    if config["data_index"] is not None:
        print(
            f"explainerdashboard ===> Generating index from column {config['data_index']}"
        )
        assert config["data_index"] in X.columns, (
            f"Cannot find data_index column ({config['data_index']})"
            f" in datafile ({config['datafile']})!"
            "Please set it to the proper index column name, or set it to null"
        )
        X = X.set_index(config["data_index"])

    params = config["params"]

    if config["explainer_type"] == "classifier":
        print(f"explainerdashboard ===> Generating ClassifierExplainer...")
        explainer = ClassifierExplainer(model, X, y, **params)
    elif config["explainer_type"] == "regression":
        print(f"explainerdashboard ===> Generating RegressionExplainer...")
        explainer = ClassifierExplainer(model, X, y, **params)
    return explainer


def build_and_dump_explainer(explainer_config, dashboard_config=None):
    explainer = build_explainer(explainer_config)

    click.echo(
        f"explainerdashboard ===> Calculating properties by building Dashboard..."
    )
    if dashboard_config is not None:
        ExplainerDashboard.from_config(explainer, dashboard_config)
    elif Path(explainer_config["explainer"]["dashboard_yaml"]).exists():
        click.echo(
            f"explainerdashboard ===> Calculating properties by building Dashboard from {explainer_config['explainer']['dashboard_yaml']}..."
        )
        dashboard_config = yaml.safe_load(
            open(str(explainer_config["explainer"]["dashboard_yaml"]), "r")
        )
        ExplainerDashboard.from_config(explainer, dashboard_config)
    else:
        click.echo(f"explainerdashboard ===> Calculating all properties")
        explainer.calculate_properties()

    click.echo(
        f"explainerdashboard ===> Saving explainer to {explainer_config['explainer']['explainerfile']}..."
    )
    if (
        dashboard_config is not None
        and explainer_config["explainer"]["explainerfile"]
        != dashboard_config["dashboard"]["explainerfile"]
    ):
        click.echo(
            f"explainerdashboard ===> Warning explainerfile in explainer config and dashboard config do not match!"
        )
    explainer.dump(explainer_config["explainer"]["explainerfile"])
    return


def launch_dashboard_from_pkl(explainer_filepath, no_browser, port, no_dashboard=False):
    explainer = BaseExplainer.from_file(explainer_filepath)

    if port is None:
        click.echo(
            f"explainerdashboard ===> Setting port to 8050, override with e.g. --port 8051"
        )
        port = 8050

    db = ExplainerDashboard(explainer, port=port)

    if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

    if not no_dashboard:
        waitress.serve(db.flask_server(), host="0.0.0.0", port=port)
    return


def launch_dashboard_from_yaml(dashboard_config, no_browser, port, no_dashboard=False):
    if isinstance(dashboard_config, (Path, str)) and str(dashboard_config).endswith(
        ".yaml"
    ):
        config = yaml.safe_load(open(str(dashboard_config), "r"))
    elif isinstance(dashboard_config, dict):
        config = dashboard_config
    else:
        raise ValueError(
            f"dashboard_config should either be a .yaml filepath or a dict!"
        )

    if not Path(config["dashboard"]["explainerfile"]).exists():
        click.echo(
            f"explainerdashboard ===> {config['dashboard']['explainerfile']} does not exist!"
        )
        click.echo(
            f"explainerdashboard ===> first generate {config['dashboard']['explainerfile']} with explainerdashboard build"
        )
        return

    click.echo(
        f"explainerdashboard ===> Building dashboard from {config['dashboard']['explainerfile']}"
    )

    db = ExplainerDashboard.from_config(config)

    if port is None:
        port = config["dashboard"]["params"]["port"]
        if port is None:
            port = 8050
        click.echo(
            f"explainerdashboard ===> Setting port to {port}, override with e.g. --port 8051"
        )

    if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
        click.echo(
            f"explainerdashboard ===> launching browser at {f'http://localhost:{port}/'}"
        )
        webbrowser.open_new(f"http://localhost:{port}/")

    click.echo(f"explainerdashboard ===> Starting dashboard:")
    if not no_dashboard:
        waitress.serve(db.flask_server(), host="0.0.0.0", port=port)
    return


def launch_hub_from_yaml(hub_config, no_browser, port, no_dashboard=False):
    hub = ExplainerHub.from_config(hub_config)

    if port is None:
        port = hub.port
        if port is None:
            port = 8050
        click.echo(
            f"explainerhub ===> Setting port to {port}, override with e.g. --port 8051"
        )

    if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
        click.echo(
            f"explainerhub ===> launching browser at {f'http://localhost:{port}/'}"
        )
        webbrowser.open_new(f"http://localhost:{port}/")

    click.echo(f"explainerhub ===> Starting dashboard:")
    if not no_dashboard:
        waitress.serve(hub.flask_server(), host="0.0.0.0", port=port)
    return


def get_hub_filepath(filepath):
    if filepath is None:
        if (Path().cwd() / "hub.yaml").exists():
            click.echo("explainerhub ===> Detected hub.yaml...")
            filepath = Path().cwd() / "hub.yaml"
        elif (Path().cwd() / "users.yaml").exists():
            click.echo("explainerhub ===> Detected users.yaml...")
            filepath = Path().cwd() / "users.yaml"
        elif (Path().cwd() / "users.json").exists():
            click.echo("explainerhub ===> Detected users.json...")
            filepath = Path().cwd() / "users.json"
        else:
            click.echo(
                "No argument given and could find neither a "
                "hub.yaml nor users.yaml or users.json. Aborting."
            )
            return

    if str(filepath).endswith(".yaml"):
        config = yaml.safe_load(open(str(filepath), "r"))
        if "explainerhub" in config:
            filepath = config["explainerhub"]["users_file"]
            click.echo(f"explainerhub ===> Using {filepath} to manage users...")
    return filepath


@click.group()
@click.pass_context
def explainerdashboard_cli(ctx):
    """
    explainerdashboard CLI tool. Used to launch an explainerdashboard from
    the commandline.

    \b
    explainerdashboard run
    ----------------------

    Run explainerdashboard and start browser directly from command line.

    \b
    Example use:
        explainerdashboard run explainer.joblib
        explainerdashboard run dashboard.yaml
        explainerdashboard run dashboard.yaml --no-browser --port 8051
        explainerdashboard run --help

    If you pass an explainer.joblib file, will launch the full default dashboard.
    Generate this file with explainer.dump("explainer.joblib")

    If you pass a dashboard.yaml file, will launch a fully configured
    explainerdashboard. Generate dashboard.yaml with
    ExplainerDashboard.to_yaml('dashboard.yaml')

    If no argument given, searches for either dashboard.yaml or
    explainer.joblib, in that order, so if you keep that naming convention
    you can simply start with:

    \b
        explainerdashboard run

    \b
    explainerdashboard build
    ------------------------

    Build and store an explainer object, based on explainer.yaml file, that indicates
    where to find stored model (e.g. model.pkl), stored datafile (e.g. data.csv),
    and other explainer parameters.

    \b
    Example use:
        explainerdashboard build explainer.yaml
        explainerdashboard build explainer.yaml dashboard.yaml
        explainerdashboard build --help

    If given a second dashboard.yaml argument, will use that dashboard
    configuration to calculate necessary properties for that specific dashboard
    configuration before storing to disk. Otherwise will use dashboard_yaml
    parameter in explainer.yaml to find configuration, or alternatively
    simply calculate all properties.

    explainer.yaml file can be generated with explainer.to_yaml("explainer.yaml")

    If no argument given, searches for explainer.yaml, so if you keep that
    naming convention you can simply start the build with:

    \b
        explainerdashboard build

    """


@explainerdashboard_cli.command(help="run dashboard and open browser")
@click.pass_context
@click.argument("explainer_filepath", nargs=1, required=False)
@click.option(
    "--no-browser",
    "-nb",
    "no_browser",
    is_flag=True,
    help="Launch a dashboard, but do not launch a browser.",
)
@click.option(
    "--port", "-p", "port", default=None, help="port to run dashboard on defaults."
)
def run(ctx, explainer_filepath, no_browser, port):
    click.echo(explainer_ascii)
    if explainer_filepath is None:
        if (Path().cwd() / "dashboard.yaml").exists():
            explainer_filepath = Path().cwd() / "dashboard.yaml"
        elif (Path().cwd() / "explainer.joblib").exists():
            explainer_filepath = Path().cwd() / "explainer.joblib"
        else:
            click.echo(
                "No argument given and could find neither a "
                "dashboard.yaml nor a explainer.joblib. Aborting."
            )
            return

    if (
        str(explainer_filepath).endswith(".joblib")
        or str(explainer_filepath).endswith(".pkl")
        or str(explainer_filepath).endswith(".pickle")
        or str(explainer_filepath).endswith(".dill")
    ):
        launch_dashboard_from_pkl(explainer_filepath, no_browser, port)
        return
    elif str(explainer_filepath).endswith(".yaml"):
        launch_dashboard_from_yaml(explainer_filepath, no_browser, port)
    else:
        click.echo(
            "Please pass a proper argument to explainerdashboard run"
            "(i.e. either an explainer.joblib or a dashboard.yaml)"
        )
    return


@explainerdashboard_cli.command(help="build and save explainer object")
@click.pass_context
@click.argument("explainer_filepath", nargs=1, required=False)
@click.argument("dashboard_filepath", nargs=1, required=False)
def build(ctx, explainer_filepath, dashboard_filepath):
    click.echo(explainer_ascii)
    if explainer_filepath is None:
        if (Path().cwd() / "explainer.yaml").exists():
            explainer_filepath = Path().cwd() / "explainer.yaml"
        else:
            click.echo(
                "No argument given to explainerdashboard build and "
                "could not find an explainer.yaml. Aborting."
            )
            return

    if str(explainer_filepath).endswith(".yaml"):
        explainer_config = yaml.safe_load(open(str(explainer_filepath), "r"))

        click.echo(
            f"explainerdashboard ===> Building {explainer_config['explainer']['explainerfile']}"
        )

        if (
            dashboard_filepath is not None
            and str(dashboard_filepath).endswith(".yaml")
            and Path(dashboard_filepath).exists()
        ):
            click.echo(
                f"explainerdashboard ===> Using {dashboard_filepath} to calculate explainer properties"
            )
            dashboard_config = yaml.safe_load(open(str(dashboard_filepath), "r"))
        else:
            dashboard_config = None

        print(
            f"explainerdashboard ===> Building {explainer_config['explainer']['explainerfile']}"
        )
        build_and_dump_explainer(explainer_config, dashboard_config)
        print(f"explainerdashboard ===> Build finished!")
        return


@explainerdashboard_cli.command(help="run without launching dashboard")
@click.pass_context
@click.argument("explainer_filepath", nargs=1, required=True)
@click.option(
    "--port", "-p", "port", default=None, help="port to run dashboard on defaults."
)
def test(ctx, explainer_filepath, port):
    if (
        str(explainer_filepath).endswith(".joblib")
        or str(explainer_filepath).endswith(".pkl")
        or str(explainer_filepath).endswith(".pickle")
        or str(explainer_filepath).endswith(".dill")
    ):
        launch_dashboard_from_pkl(
            explainer_filepath, no_browser=True, port=port, no_dashboard=True
        )
        return

    elif str(explainer_filepath).endswith(".yaml"):
        launch_dashboard_from_yaml(
            explainer_filepath, no_browser=True, port=port, no_dashboard=True
        )
        return
    else:
        raise ValueError(
            "Please pass a proper argument " "(i.e. .joblib, .pkl, .dill or .yaml)!"
        )


@click.group()
@click.pass_context
def explainerhub_cli(ctx):
    """
    explainerhub CLI tool. Used to launch and manage explainerhub from
    the commandline.

    \b
    explainerhub run
    ----------------------

    Run explainerdashboard and start browser directly from command line.

    \b
    Example use:
        explainerhub run hub.yaml

    \b
    If no argument given assumed argument is hub.yaml

    \b
    explainerhub user management
    ----------------------------

    You can use the CLI to add and remove users from the users.json file that
    stores the usernames and (hashed) passwords for the explainerhub. If no
    filename is given, will look for either a hub.yaml or users.json file.

    \b
    If you don't provide the username or password on the commandline, you will get prompted.

    \b
    Examples use:
        explainerhub add-user
        explainerhub add-user users.yaml
        explainerhub add-user users2.json
        explainerhub add-user hub.yaml

        explainerhub delete-user
        explainerhub add-dashboard-user
        explainerhub delete-dashboard-user

    """


@explainerhub_cli.command(help="run explainerhub and open browser")
@click.pass_context
@click.argument("hub_filepath", nargs=1, required=False)
@click.option(
    "--no-browser",
    "-nb",
    "no_browser",
    is_flag=True,
    help="Launch hub, but do not launch a browser.",
)
@click.option("--port", "-p", "port", default=None, help="port to run hub on.")
def run(ctx, hub_filepath, no_browser, port):
    if hub_filepath is None:
        if (Path().cwd() / "hub.yaml").exists():
            hub_filepath = Path().cwd() / "hub.yaml"
        else:
            click.echo(
                "No argument given and could find neither a " "hub.yaml. Aborting."
            )
            return
    click.echo(hub_ascii)
    launch_hub_from_yaml(hub_filepath, no_browser, port)


@explainerhub_cli.command(help="add a user to users.yaml")
@click.argument("filepath", nargs=1, required=False)
@click.option("--username", "-u", required=True, prompt=True)
@click.option(
    "--password",
    "-p",
    required=True,
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
)
def add_user(filepath, username, password):
    filepath = get_hub_filepath(filepath)
    if filepath is None:
        return
    ExplainerHub._validate_users_file(filepath)
    ExplainerHub._add_user_to_file(filepath, username=username, password=password)
    click.echo(f"explainerhub ===> user added to {filepath}!")


@explainerhub_cli.command(help="remove a user from users.yaml")
@click.argument("filepath", nargs=1, required=False)
@click.option("--username", "-u", required=True, prompt=True)
def delete_user(filepath, username):
    filepath = get_hub_filepath(filepath)
    if filepath is None:
        return

    ExplainerHub._validate_users_file(filepath)
    ExplainerHub._delete_user_from_file(filepath, username=username)
    click.echo(f"explainerhub ===> user removed from {filepath}!")


@explainerhub_cli.command(help="add a username to a dashboard users.yaml")
@click.argument("filepath", nargs=1, required=False)
@click.option("--dashboard", "-d", required=True, prompt=True)
@click.option("--username", "-u", required=True, prompt=True)
def add_dashboard_user(filepath, dashboard, username):
    filepath = get_hub_filepath(filepath)
    if filepath is None:
        return
    ExplainerHub._validate_users_file(filepath)
    ExplainerHub._add_user_to_dashboard_file(
        filepath, dashboard=dashboard, username=username
    )
    click.echo(f"explainerhub ===> user added to {dashboard} in {filepath}!")


@explainerhub_cli.command(help="remove a username from a dashboard in users.yaml")
@click.argument("filepath", nargs=1, required=False)
@click.option("--dashboard", "-d", required=True, prompt=True)
@click.option("--username", "-u", required=True, prompt=True)
def delete_dashboard_user(filepath, dashboard, username):
    filepath = get_hub_filepath(filepath)
    if filepath is None:
        return
    ExplainerHub._validate_users_file(filepath)
    ExplainerHub._delete_user_from_dashboard_file(
        filepath, dashboard=dashboard, username=username
    )
    click.echo(f"explainerhub ===> user removed from {dashboard} in {filepath}!")


if __name__ == "__main__":
    explainerdashboard_cli()
