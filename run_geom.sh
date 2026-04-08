#!/usr/bin/env bash

python3 3_characterize_hf_dataset.py \
  --dataset "descent_format_geom" \
  --card-pretty-name "Meta-OMol25 Descent Formatted GEOM v1.0" \
  --metadata-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom/geom_orca6_metadata" \
  --smee-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom/geom_orca6_smee" \
  --failed-rows-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom/geom_orca6_failed_rows.jsonl" \
  --ase-db-path "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/ase_databases/geom_orca6.db" \
  --output-dir "/Users/jenniferclark/OMSF/OpenFF/DatasetRelease/MetaOMol25_org_smiles/geom" \
  --sample-size -1 \
  --processability-rows 256 \
  --strict-index-max -1

#python 4_push_to_hub.py \
#  --repo-id openforcefield/descent-format-geom \
#  --metadata-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom_orca6_metadata \
#  --smee-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom_orca6_smee \
#  --metadata-config metadata \
#  --smee-config descent_data