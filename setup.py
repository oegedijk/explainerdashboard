from setuptools import setup, find_packages

setup(
    name='explainerbunch',
    version='0.0.2',
    description='package to ease and speed up machine learning model explainability',
    license='MIT',
    packages=find_packages(),
    package_data={
        'dashboard': ['*.css', '*.min.js'],
    },
    install_requires=['dash', 'dash-daq', 'dash-bootstrap-components',
                    'dtreeviz', 'numpy', 'pandas', 'PDPbox', 'scikit-learn', 'shap'],
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning', 'explainability', 'shap', 'feature importances', 'dash'],
    url='https://github.com/oegedijk/explainerbunch'
)