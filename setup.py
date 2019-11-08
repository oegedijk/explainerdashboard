from setuptools import setup, find_packages

setup(
    name='explainerdashboard',
    version='0.1',
    description='package to ease and speed up machine learning model explainability',
    long_description='explainerdashboard allows you quickly build an interactive dashboard to explain the inner workings of a machine learning model.',
    license='MIT',
    packages=find_packages(),
    package_dir={'explainerdashboard':'explainerdashboard'}, # the one line where all the magic happens
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
    install_requires=['dash', 'dash-daq', 'dash-bootstrap-components',
                    'dtreeviz', 'numpy', 'pandas', 'PDPbox', 'scikit-learn', 'shap'],
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning', 'explainability', 'shap', 'feature importances', 'dash'],
    url='https://github.com/oegedijk/explainerdashboard'
)