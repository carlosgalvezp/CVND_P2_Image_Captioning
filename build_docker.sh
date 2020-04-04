#!/usr/bin/env bash
set -euxo pipefail

cp requirements.txt docker
docker build docker
rm docker/requirements.txt
