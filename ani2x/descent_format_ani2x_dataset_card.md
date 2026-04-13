---
pretty_name: Meta-OMol25 Descent Formatted ANI2X v1.0
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

# Dataset Card for Meta-OMol25 Descent Formatted ANI2X v1.0

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

- Meta-OMol25 rows: 9641964
- Metadata rows: 9323426
- Smee rows: 9323426
- Failed rows: 318538
- Failed unique indices: 318538
- Accounted total (metadata + failed - overlap): 9641964
- All Meta-OMol25 structures accounted for: True
- Uncovered Meta-OMol25 rows: 0

### Metadata Split and Chemistry Summary

- Sample size used for summary: 9323426
- Sample split counts: {
  "val": 2861,
  "train": 9320565
}
- Sample charge range: [0, 0]
- Sample molecular weight (g/mol, min / max / mean): 16.043000 / 408.458000 / 122.197604
- Molecular weight sample count: 9323426

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
- Min: -2065007.125000
- Max: -25377.093750
- Mean: -367128.856687

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
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry R.'): 47184
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry S.'): 43965
- ValueError('Inconsistent charge with target!'): 5567
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 2 radical electrons on molecule [H][O+].[O-].[O]Cl.'): 278
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H].[H].[H]C([H])([H])C([N-])=[O+].'): 256
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 2 radical electrons on molecule [H][C][C@@]12N3[C@@]([H])(C1([H])[H])[C@@]3([H])C([H])([H])C2([H])[H].'): 231
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 2 radical electrons on molecule [H+].[H+].[H+].[H]N([H])C([H])([H])[C][C][C-3].'): 228
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry Z.'): 219
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry E.'): 209
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H].[H].[H]C([H])=C([H])C([N-])=[O+].'): 173

## Dataset Card Authors

- Jennifer A Clark (Open Force Field Initiative); jaclark5

## Dataset Card Contact

- Primary contact: info@openforcefield.org
