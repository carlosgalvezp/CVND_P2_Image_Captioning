#!/usr/bin/env bash
set -euxo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip3 install wheel==0.34.2  # Required to install jupyter later on
pip3 install -r requirements.txt
