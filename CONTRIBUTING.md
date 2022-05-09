# Contributing

## Runnning tests offline

When submitting a PR github can run the full testsuite using github actions. For running the tests offline you
need to make sure you have installed all the required offline testing requirements:

### virtual environment

First create a new virtual environment:

`$ python -m venv venv`
`$ source venv/bin/activate`

### install dependencies and CLI tools

First make sure you have the latest version of pip itself: 
`$ python -m pip install -U pip setuptools wheel`

Then install the whole package including dependencies:
`$ pip install -e .`

(this also install the CLI tools in the path)

### install testing dependencies

There are additional libraries such as selenium, xgboost, catboost, lightgbm etc needed for testing:

`$ pip install -r requirements_testing.txt`

(lightgbm may give some headaches when installing with pip, so can also `brew install lightgbm` instead)

### install chromedriver for integration tests

For the integration tests we use Selenium which launches a headless version of google chrome to launch a dashboard
in the browser and then checks that there are no error messages. In order to run these tests you need to download
a chromedriver that is compatible with your current installation of chrome at https://chromedriver.chromium.org/

You then unzip it and copy it to `$ cp chromedriver /usr/local/bin/chromedriver`
and on OSX allow it to be run with `$ xattr -d com.apple.quarantine /usr/local/bin/chromedriver`.

### running the tests

The tests should now run in the base directory with

`$ pytest .`

### Skipping selenium and cli test

If you would like to skip the abovementioned selenium based integration tests, you can skip all tests marked 
(i.e. labeled with pytest.mark) with `selenium` by running e.g.:

```sh
$ pytest . -m "not selenium"
```

for also skipping all cli tests, run

```sh
$ pytest . -m "not selenium and not cli"