#!/usr/bin/env bash

# Step 2 run on HPC

python3 ../3_characterize_hf_dataset.py \
  --dataset "descent_format_ani2x" \
  --card-pretty-name "Meta-OMol25 Descent Formatted ANI2X v1.0" \
  --metadata-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_metadata" \
  --smee-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_smee" \
  --failed-rows-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_failed_rows.jsonl" \
  --ase-db-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/ase_databases/ani2x.db" \
  --output-dir "/Users/jenniferclark/OMSF/OpenFF/DatasetRelease/MetaOMol25_org_smiles/ani2x" \
  --sample-size -1 \
  --processability-rows 256 \
  --strict-index-max -1

python ../4_push_to_hub.py \
  --repo-id openforcefield/descent-format-ani2x \
  --metadata-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_metadata \
  --smee-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_smee \
  --metadata-config metadata \
  --smee-config descent_data