#!/usr/bin/env bash
set -euo pipefail

# Generate API rst files from the source package.
python -m sphinx.ext.apidoc \
  -o docs/source/api \
  src/jax_sbgeom \
  --force \
  --module-first \
  --separate \
  -d 2

# Build HTML docs.
python -m sphinx -b html docs/source docs/build/html
