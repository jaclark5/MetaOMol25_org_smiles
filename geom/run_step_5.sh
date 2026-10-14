#!/usr/bin/env bash

mkdir -p quality_control
python3 ../5_smiles_quality.py \
  --repo-id openforcefield/descent-format-geom \
  --max-rows -1 \
  --output-dir "quality_control" > quality_control/log.txt
