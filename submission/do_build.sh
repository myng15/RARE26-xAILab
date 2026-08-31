#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="rare26-dinov2-lora-attnmil-ensemble-xailab"

# --provenance=false --sbom=false: emit a single-platform image, not a manifest list with
# attestations, so `docker save` produces a tarball Grand Challenge imports cleanly.
docker build \
  --platform=linux/amd64 \
  --provenance=false \
  --sbom=false \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR" 2>&1
