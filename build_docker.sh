#!/usr/bin/env bash
set -euxo pipefail

export DOCKER_BUILDKIT=1

cp requirements.txt docker/
docker build -t carlosgalvezp/cvnd_p2_image_captioning:latest docker/
rm docker/requirements.txt
