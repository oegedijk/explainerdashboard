"""
Command-line tool for starting an explainerdashboard from a particular directory
"""
import os
import webbrowser
from pathlib import Path
import pickle
import yaml

import pandas as pd
import joblib
import click

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.explainers import BaseExplainer
from explainerdashboard.dashboards import ExplainerDashboard


def build_explainer(explainer_config):
    model = pickle.load(open(explainer_config['modelfile'], "rb"))
    df = pd.read_csv(explainer_config['datafile'])
    target_col = explainer_config['data_target']
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if explainer_config['data_index'] is not None:
        if explainer_config['data_index'] == 'index':
            idxs = pd.Series(X.index)
        else:
            idxs = X[explainer_config['data_index']]
    else:
        idxs = None
        
    params = explainer_config['params']

    if explainer_config['explainer_type'] == "classifier":
        explainer = ClassifierExplainer(model, X, y, idxs=idxs, **params)
    elif explainer_config['explainer_type'] == "regression":
        explainer = ClassifierExplainer(model, X, y, idxs=idxs, **params)
    return explainer


def build_and_dump_explainer(config):
    explainer = build_explainer(config['explainer'])
    db = ExplainerDashboard(explainer, **config['dashboard']['params'])
    explainer.dump(config['explainer']['explainerfile'])
    return


@click.command()
@click.argument("explainer_filepath", nargs=1, required=False)
@click.option("--build-only", "-bo", "build_only", is_flag=True,
                help="Do not launch a dashboard, only build the explainer an store it.")
@click.option("--no-browser", "-nb", "no_browser", is_flag=True,
                help="Launch a dashboard, but do not launch a browser.")
@click.option("--port", "-p", "port", default=None,
                help="port to run dashboard on defaults")
def run_dashboard(explainer_filepath, build_only, no_browser, port):
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

        db.run(port)
        return

    elif (str(explainer_filepath).endswith(".yaml") or
          str(explainer_filepath).endswith(".yml")):
        config = yaml.safe_load(open(str(explainer_filepath), "r"))
        if build_only:
            print(f"Building {config['explainer']['explainerfile']}")
            build_and_dump_explainer(config)
            print(f"Build finished!")
            return 

        if not Path(config['dashboard']['explainer']).exists():
            build_and_dump_explainer(config)

        explainer = BaseExplainer.from_file(config['dashboard']['explainer'])

        db = ExplainerDashboard(explainer, **config['dashboard']['params'])

        if port is None: 
            port =  config['dashboard']['params']['port'] 
            if port is None:
                port = 8050

        if not no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f"http://localhost:{port}/")
        
        db.run(port)
        return

          



if __name__ =="__main__":
    run_dashboard()

