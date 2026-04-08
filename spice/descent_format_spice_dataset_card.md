---
pretty_name: Meta-OMol25 Descent Formatted SPICE v1.0
license: CC-BY-4.0
task_categories:
- chemistry
language:
- en
tags:
- openff
- molecular-mechanics
- force-field-fitting
- vdw
- valence
---

# Dataset Card for Meta-OMol25 Descent Formatted SPICE v1.0

## Dataset Details

### Dataset Description

Meta-OMol25 provides molecular structures, coordinates, energies, and forces, and we derived mapped SMILES for broad OpenFF parameter fitting workflows. This release is designed for general fitting and evaluation of van der Waals and valence terms.

- Curated by: Jennifer A Clark; jaclark5
- Funded by: Open Force Field Initiative
- Shared by: Open Force Field Initiative, Open Molecular Software Foundation
- License: CC-BY-4.0
- Dataset version: v1.0

### Dataset Sources

- Repository: https://huggingface.co/facebook/OMol25
- Hugging Face dataset repository: {{HF_REPO_URL}}

## Uses

### Direct Use

- Fit or benchmark van der Waals parameters.
- Fit or benchmark valence terms (bonds, angles, torsions).
- Provide aligned molecular metadata and per-structure coordinates / energies / forces for OpenFF workflows.

## Dataset Structure

### Overall Statistics

- Meta-OMol25 rows: 1985965
- Metadata rows: 1956538
- Smee rows: 1956538
- Failed rows: 29427
- Failed unique indices: 29427
- Accounted total (metadata + failed - overlap): 1985965
- All Meta-OMol25 structures accounted for: True
- Uncovered Meta-OMol25 rows: 0

### Metadata Split and Chemistry Summary

- Sample size used for summary: 1956538
- Sample split counts: {
  "val": 9554,
  "train": 1946984
}
- Sample charge range: [-5, 3]
- Sample molecular weight (g/mol, min / max / mean): 4.032000 / 1451.552762 / 285.915873
- Molecular weight sample count: 1956538

### Metadata Schema

- OMol25_data_id: Value('string')
- OMol25_id: Value('int64')
- OMol25_split: Value('string')
- OpenFF_id: Value('int64')
- formula: Value('string')
- charge: Value('int64')
- OpenFF_Elements: Value('bool')
- OpenFF_abs(q)<=1: Value('bool')
- OpenFF_spin=1: Value('bool')
- smiles: Value('string')
- source: Value('string')
- lowdin_charges: List(Value('float64'))

### Smee Schema

- smiles: Value('string')
- coords: List(Value('float32'))
- energy: List(Value('float32'))
- forces: List(Value('float32'))

Sample energy stats (kcal/mol):
- Min: -17642260.000000
- Max: -1401.253174
- Mean: -856549.566531

### Cross-Table Consistency

- Metadata and smee row count match: True
- Sample metadata/smee SMILES mismatches: 0

## Dataset Creation

### Curation Rationale

- This dataset is intended for broad force-field parameterization workloads and diagnostics.
- This release emphasizes direct support for van der Waals and valence fitting tasks.

### Data Collection and Processing

- Upstream rows are converted into metadata and smee datasets.
- Rows that fail conversion are logged in failed_rows JSONL and used for reconciliation.
- Coverage checks validate whether all Meta-OMol25 rows are represented by either successful records or failed-row entries.

### Quality and Validation

- smee importable: True
- descent importable: True
- descent create_dataset on sample succeeded: True
- Validation sample size: 256
- Validation error (if any): 

Top failure modes:
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry S.'): 6777
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry R.'): 6156
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H][O+][H].[K].'): 2845
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H][O+][H].[Na].'): 2745
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H][O+][H].[Li].'): 2677
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [Ca+].[H][O+][H].'): 1411
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H][O+][H].[Mg+].'): 1411
- ValueError('Inconsistent charge with target!'): 268
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry E.'): 84
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry Z.'): 79

## Dataset Card Authors

- Jennifer A Clark (Open Force Field Initiative); jaclark5

## Dataset Card Contact

- Primary contact: info@openforcefield.org
