#!/usr/bin/env bash

python3 3_characterize_hf_dataset.py \
  --dataset "descent_format_spice" \
  --card-pretty-name "Meta-OMol25 Descent Formatted SPICE v1.0" \
  --metadata-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/spice_metadata" \
  --smee-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/spice_smee" \
  --failed-rows-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/spice_failed_rows.jsonl" \
  --ase-db-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/ase_databases/spice.db" \
  --output-dir "/Users/jenniferclark/OMSF/OpenFF/DatasetRelease/MetaOMol25_org_smiles/spice" \
  --sample-size -1 \
  --processability-rows 256 \
  --strict-index-max -1

#python 4_push_to_hub.py \
#  --repo-id openforcefield/descent-format-spice \
#  --metadata-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/spice/spice_metadata \
#  --smee-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/spice/spice_smee \
#  --metadata-config metadata \
#  --smee-config descent_data