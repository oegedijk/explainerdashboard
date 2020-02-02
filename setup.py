from setuptools import setup, find_packages

setup(
    name='explainerdashboard',
    version='0.1.8.2',
    description='explainerdashboard allows you quickly build an interactive dashboard to explain the inner workings of your machine learning model.',
    long_description="""
explainerdashboard allows you quickly build an interactive dashboard to explain the inner workings of your machine learning model.
The library is flexible in that you first create an ExplainerBunch class that handles the computations and 
plotting functionality for you, so that you can then build a plotly dash dashboard on top of that. 

The standard built-in dashboard comes with a number of standard tabs (that can be switched on individually), namely:

- Model Summary Tab (classifier/regression metrics and plots + feature importances)
- Contributions Tab (explain individual predictions, and compare what-if scenarios using pdp plots)
- Dependence Tab (investigate how predictions change along the axis of each feature)
- Interactions Tab (investigate the interaction effects between your variables)
- Shadow Trees Tab (for RandomForests, display all the individual trees inside the forest)

It includes:
- Model summary statistics
- SHAP values (importances, individual contributions, dependence plots, interaction plots, etc)
- permutation importances
- partial dependence plots
- DecisionTree visualizers

You can display an interactive dashboard with all of these features with only three lines of code.

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
    install_requires=['dash', 'dash-bootstrap-components',
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