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

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.explainers import BaseExplainer
from explainerdashboard.dashboards import ExplainerDashboard


explainer_ascii = """

                 _       _                    _           _     _                         _ 
                | |     (_)                  | |         | |   | |                       | |
  _____  ___ __ | | __ _ _ _ __   ___ _ __ __| | __ _ ___| |__ | |__   ___   __ _ _ __ __| |
 / _ \ \/ | '_ \| |/ _` | | '_ \ / _ | '__/ _` |/ _` / __| '_ \| '_ \ / _ \ / _` | '__/ _` |
|  __/>  <| |_) | | (_| | | | | |  __| | | (_| | (_| \__ | | | | |_) | (_) | (_| | | | (_| |
 \___/_/\_| .__/|_|\__,_|_|_| |_|\___|_|  \__,_|\__,_|___|_| |_|_.__/ \___/ \__,_|_|  \__,_|
          | |                                                                               
          |_|                                                                               
"""

def build_explainer(explainer_config):
    print(f"explainerdashboard ===> Loading model from {explainer_config['modelfile']}")
    model = pickle.load(open(explainer_config['modelfile'], "rb"))
    print(f"explainerdashboard ===> Loading data from {explainer_config['datafile']}")
    df = pd.read_csv(explainer_config['datafile'])

    print(f"explainerdashboard ===> Using column {explainer_config['data_target']} to generate X, y ")
    target_col = explainer_config['data_target']
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    if explainer_config['data_index'] is not None:
        print(f"explainerdashboard ===> Generating index from column {explainer_config['data_index']}")
        assert explainer_config['data_index'] in X.columns, \
            (f"Cannot find data_index column ({explainer_config['data_index']})"
             f" in datafile ({explainer_config['datafile']})!"
              "Please set it to the proper index column name, or set it to null")
        X = X.set_index(explainer_config['data_index'])
        
    params = explainer_config['params']

    if explainer_config['explainer_type'] == "classifier":
        print(f"explainerdashboard ===> Generating ClassifierExplainer...")
        explainer = ClassifierExplainer(model, X, y, **params)
    elif explainer_config['explainer_type'] == "regression":
        print(f"explainerdashboard ===> Generating RegressionExplainer...")
        explainer = ClassifierExplainer(model, X, y, **params)
    return explainer


def build_and_dump_explainer(config):
    explainer = build_explainer(config['explainer'])

    print(f"explainerdashboard ===> Calculating properties by building Dashboard...")
    dashboard_params = deepcopy(config['dashboard']['params'])
    tabs = yamltabs_to_tabs(dashboard_params['tabs'])
    del dashboard_params['tabs']
    db = ExplainerDashboard(explainer, tabs, **dashboard_params)

    print(f"explainerdashboard ===> Saving explainer to {config['explainer']['explainerfile']}...")
    explainer.dump(config['explainer']['explainerfile'])
    return


def yamltabs_to_tabs(tabs_param):
    """converts a yaml tabs list back to ExplainerDashboard compatible original"""
    if tabs_param is None:
        return None
    
    if not hasattr(tabs_param, "__iter__"):
        if isinstance(tabs_param, str):
            return tabs_param
        return getattr(import_module(tabs_param['module']), tabs_param['tab']) 
        
    return [tab if isinstance(tab, str) else getattr(import_module(tab['module']), tab['tab'])  for tab in tabs_param]


@click.command()
@click.argument("explainer_filepath", nargs=1, required=False)
@click.option("--build-only", "-bo", "build_only", is_flag=True,
                help="Do not launch a dashboard, only build the explainer and store it.")
@click.option("--no-browser", "-nb", "no_browser", is_flag=True,
                help="Launch a dashboard, but do not launch a browser.")
@click.option("--no-dashboard", "-nd", "no_dashboard", is_flag=True,
                help="Do not launch dashboard in the end. Used for testing.")
@click.option("--port", "-p", "port", default=None,
                help="port to run dashboard on defaults.")
def run_dashboard(explainer_filepath, build_only, no_browser, no_dashboard, port):
    print(explainer_ascii)
    if explainer_filepath is None:
        if (Path().cwd() / "explainerdashboard.yaml").exists():
            explainer_filepath = Path().cwd() / "explainerdashboard.yaml"
        elif (Path().cwd() / "explainer.joblib").exists():
            explainer_filepath = Path().cwd() / "explainer.joblib"
        else:
            raise ValueError("Could not find a explainerdashboard.yaml nor a "
                    "explainer.joblib and you didn't pass an argument. ")
    if (str(explainer_filepath).endswith(".joblib") or
        str(explainer_filepath).endswith(".pkl") or
        str(explainer_filepath).endswith(".pickle") or
        str(explainer_filepath).endswith(".joblib")):
        explainer = BaseExplainer.from_file(explainer_filepath)

        if port is None: 
            port = 8050

        db = ExplainerDashboard(explainer, port=port)
        
        if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f"http://127.0.0.1:{port}/")

        if not no_dashboard:
            db.run(port)
        return

    elif (str(explainer_filepath).endswith(".yaml") or
          str(explainer_filepath).endswith(".yml")):
        config = yaml.safe_load(open(str(explainer_filepath), "r"))
        if build_only:
            print(f"explainerdashboard ===> Building {config['explainer']['explainerfile']}")
            build_and_dump_explainer(config)
            print(f"explainerdashboard ===> Build finished!")
            return 

        if not Path(config['dashboard']['explainerfile']).exists():
            print(f"explainerdashboard ===> {config['dashboard']['explainerfile']} does not exist!")
            if (
                config['explainer']['explainerfile'] == config['dashboard']['explainerfile'] and
                Path(config['explainer']['modelfile']).exists() and
                Path(config['explainer']['datafile']).exists()
                ):
                click.confirm(f"Do you want to try and build {config['dashboard']['explainerfile']} from scratch?", abort=True)
                print(f"explainerdashboard ===> Attempting to build {config['explainer']['explainerfile']}")
                try:
                    build_and_dump_explainer(config)
                except:
                    print(f"explainerdashboard ===> Build failed:")
                    raise

        print(f"explainerdashboard ===> Building dashboard from {config['dashboard']['explainerfile']}")
        explainer = BaseExplainer.from_file(config['dashboard']['explainerfile'])

        print(f"explainerdashboard ===> Building dashboard")

        dashboard_params = config['dashboard']['params']
        tabs = yamltabs_to_tabs(dashboard_params['tabs'])
        del dashboard_params['tabs']

        db = ExplainerDashboard(explainer, tabs, **dashboard_params)

        if port is None: 
            port =  config['dashboard']['params']['port'] 
            if port is None:
                port = 8050

        if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
            print(f"explainerdashboard ===> launching browser at {f'http://localhost:{port}/'}")
            webbrowser.open_new(f"http://localhost:{port}/")
        
        print(f"explainerdashboard ===> Starting dashboard:")
        if not no_dashboard:
            db.run(port)
        return

          



if __name__ =="__main__":
    run_dashboard()

