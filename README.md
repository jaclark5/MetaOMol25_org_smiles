# MetaOMol25 OpenFF Dataset Builder

This repository is a four-step pipeline for building OpenFF-ready Hugging Face datasets from Meta-OMol25.

- Step 1: prepare/filter ASE databases with 1_sort_data.ipynb
- Step 2: convert ASE data into sharded + merged Hugging Face datasets with 2_get_smiles_sharded.py
- Step 3: characterize dataset quality and processability with 3_characterize_hf_dataset.py
- Step 4: push metadata + spice configs to Hugging Face Hub with 4_push_to_hub.py

The conversion step is designed to:

- Stream structures from an ASE SQLite database
- Convert each structure to OpenFF-compatible records
- Save metadata and smee outputs in shards
- Resume safely from checkpoints
- Merge shard sets into single metadata and single smee Hugging Face datasets after processing

## Repository Contents

- 1_sort_data.ipynb: Step 1 notebook. Builds filtered ASE databases from train/val OMol25 data and writes split-aware metadata used by step 2.
- 2_get_smiles_sharded.py: Step 2 script. Converts ASE DB rows to OpenFF metadata+smee Hugging Face datasets.
- 3_characterize_hf_dataset.py: Step 3 script. Generates characterization reports and a dataset card draft.
- 4_push_to_hub.py: Step 4 script. Pushes local metadata and spice (or smee) datasets as Hub configs.
- geom/: Additional project data
- notes.txt, breakdown.xlsx: Project notes and analysis artifacts
- LICENSE: License file

## Workflow

1. Run 1_sort_data.ipynb

- Download/extract train and val sources if needed
- Filter target data IDs (for example, geom_orca6)
- Write filtered ASE databases under:
  dataset-root/ase_databases/<ds-name>.db

2. Run 2_get_smiles_sharded.py

- Read from ASE DB created in step 1
- Convert to OpenFF-compatible records
- Write metadata/smee shards plus merged metadata/smee datasets

3. Run 3_characterize_hf_dataset.py

- Summarize metadata/smee dataset quality
- Validate processability with descent/smee
- Write JSON, text, and dataset card outputs

4. Run 4_push_to_hub.py

- Load local metadata/spice (or metadata/smee) datasets from disk
- Push each dataset to the same Hub repo using separate `config_name` values

## Requirements

Use a Python environment with packages compatible with your local workflow.

Core imports used by the script:

- ase
- numpy
- scipy
- datasets
- openff-toolkit
- descent
- tmos

You also need access to:

- A Meta-OMol25 ASE database file at:
  dataset-path/ase_databases/<ds-name>.db

## How It Works

This section describes step 2 (2_get_smiles_sharded.py).

For each shard:

1. Read rows from ASE DB using offset and limit
2. Build per-row payloads
3. Convert payloads in parallel with worker threads
4. Save two shard outputs:
   - metadata shard
   - smee shard
5. Append failed rows to a JSONL log
6. Update checkpoint for resumability

After all shards are processed, the script streaming-merges:

- metadata shards into one merged metadata dataset
- smee shards into one merged smee dataset

This merge strategy avoids loading all shards into memory at once.

## CLI Usage

All of the following arguments are currently required in practice:

- --dataset-path
- --output-path
- --ds-name

Optional arguments:

- --shard-size (default: 250)
- --workers (default: min(4, cpu_count), minimum 1)
- --report-every (default: 10000)

Example:

    python 2_get_smiles_sharded.py \
      --dataset-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25 \
      --output-path /Volumes/jenclar/OpenFF \
      --ds-name geom_orca6 \
      --shard-size 250 \
      --workers 4 \
      --report-every 10000

Step 4 example:

    python 4_push_to_hub.py \
      --repo-id your-username/your-dataset \
      --metadata-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom_orca6_metadata \
      --smee-path /Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/geom_orca6_smee \
      --metadata-config metadata \
      --smee-config smee

## Output Layout

The script writes under:

- output-path/huggingface_datasets

Expected outputs:

- <ds-name>_metadata_shards/
- <ds-name>_smee_shards/
- <ds-name>_metadata
- <ds-name>_smee
- <ds-name>_checkpoint.json
- <ds-name>_failed_rows.jsonl

## Resume Behavior

If checkpoint exists, the script resumes from checkpoint next_offset.

Shard-level skip logic:

- A shard is considered complete only if both corresponding metadata and smee shard directories exist.

This makes restart behavior robust after interruptions.

## Performance and Memory Notes

- Shard processing bounds memory growth versus one-shot dataset creation, it's expected that 100 structures need 235 MB.
- Smaller shard sizes reduce peak memory but increase overhead.
- Larger shard sizes improve throughput but can raise memory usage.
- Worker count should be tuned per machine.

Practical tuning suggestions:

- Start with workers 2 to 4
- Increase shard-size gradually if memory headroom is available
- Keep report-every at a moderate value to avoid noisy logs

## Failure Logging

Rows that fail both conversion methods are appended to:

- <ds-name>_failed_rows.jsonl

Each entry includes:

- index
- ase_id
- error

## Troubleshooting

1. Script exits immediately with argparse error

- Ensure dataset-path, output-path, and ds-name are provided.

2. Conversion is slow

- Check workers setting and storage path performance.
- Use SSD for output-path when possible.

3. High failure count

- Inspect failed_rows JSONL to identify dominant error classes.

4. Merge already exists

- If merged metadata or smee dataset already exists, merge step is skipped.

## Safety Notes

- Keep output-path on a filesystem with sufficient free space.
- Do not delete checkpoint or shard directories mid-run.
- If moving outputs between disks, preserve directory structure and checkpoint files.
