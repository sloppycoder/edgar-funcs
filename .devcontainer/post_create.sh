#!/bin/bash

# this project needs these pakcages to work
# sudo apt update && sudo apt install -y --no-install-recommends redis

uv sync
uv run pre-commit install
