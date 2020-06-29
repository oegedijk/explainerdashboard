from setuptools import setup, find_packages

setup(
    name='explainerdashboard',
    version='0.1.13.0',
    description='explainerdashboard allows you quickly build an interactive dashboard to explain the inner workings of your machine learning model.',
    long_description="""

    This package makes it convenient to quickly explain the workings of a (scikit-learn compatible) fitted machine learning model using either interactive plots in e.g. Jupyter Notebook or deploying an interactive dashboard (based on Flask/Dash) that allows you to quickly explore the impact of different features on model predictions. Example deployed at: titanicexplainer.herokuapp.com

In a lot of organizations, especially governmental, but with the GDPR also increasingly in private sector, it becomes more and more important to be able to explain the inner workings of your machine learning algorithms. Customers have to some extent a right to an explanation why they were selected, and more and more internal and external regulators require it. With recent innovations in explainable AI (e.g. SHAP values) the old black box trope is nog longer valid, but it can still take quite a bit of data wrangling and plot manipulation to get the explanations out of a model. This library aims to make this easy.

The goal is manyfold:

    - Make it easy for data scientists to quickly inspect the workings and performance of their model in a few lines of code
    - Make it possible for non data scientist stakeholders such as managers, directors, internal and external watchdogs to interactively inspect the inner workings of the model without having to depend on a data scientist to generate every plot and table
    - Make it easy to build an application that explains individual predictions of your model for customers that ask for an explanation
    - Explain the inner workings of the model to the people working with so that they gain understanding what the model does and doesn't do. This is important so that they can gain an intuition for when the model is likely missing information and may have to be overruled.

The library includes:

    - Shap values (i.e. what is the contributions of each feature to each individual prediction?)
    - Permutation importances (how much does the model metric deteriorate when you shuffle a feature?)
    - Partial dependence plots (how does the model prediction change when you vary a single feature?
    - Shap interaction values (decompose the shap value into a direct effect an interaction effects)
    - For Random Forests: what is the prediction of each individual decision tree, and what is the path through each tree? (using dtreeviz)
    - Plus for classifiers: precision plots, confusion matrix, ROC AUC plot, PR AUC plot, etc
    - For regression models: goodness-of-fit plots, residual plots, etc.

The library is designed to be modular so that it should be easy to design your own interactive dashboards with plotly dash, with most of the work of calculating and formatting data, and rendering plots and tables handled by explainerdashboard, so that you can focus on the layout, logic of the interactions, and project specific textual explanations of the dashboard. (i.e. design it so that it will be interpretable for business users in your organization, not just data scientists)

Alternatively, there is a built-in standard dashboard with pre-built tabs that you can select individually.


A deployed example can be found at http://titanicexplainer.herokuapp.com
""",
    license='MIT',
    packages=find_packages(),
    package_dir={'explainerdashboard': 'explainerdashboard'}, # the one line where all the magic happens
    package_data={
        'explainerdashboard': ['assets/*', 'datasets/*'],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Web Environment",
        "Framework :: Dash",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    install_requires=['dash', 'dash-bootstrap-components', 'jupyter_dash',
                    'dtreeviz', 'numpy', 'pandas', 'PDPbox', 'scikit-learn', 'shap'],
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning', 'explainability', 'shap', 'feature importances', 'dash'],
    url='https://github.com/oegedijk/explainerdashboard',
    project_urls={
        "Github page": "https://github.com/oegedijk/explainerdashboard/",
        "Documentation": "https://explainerdashboard.readthedocs.io/",
    },
)
