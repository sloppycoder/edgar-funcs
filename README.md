
# Welcome to Python project

## Cookiecutter template for Python3 project

This is a [cookiecutter](https://www.cookiecutter.io/) template for generic Python3 project with preconfigured with the following tools:

* [uv](https://docs.astral.sh/uv/)
* [ruff](https://docs.astral.sh/ruff/)
* [pre-commit](https://pre-commit.com/)
* [pyright](https://github.com/microsoft/pyright)
* VS Code support

## Setup

The easiest way to get started is use [Visual Studio Code with devcontainer](https://code.visualstudio.com/docs/devcontainers/containers)


```shell

# create virtualenv and install dependencies
uv sync
source .venv/bin/activate
ruff check --fix .

```
