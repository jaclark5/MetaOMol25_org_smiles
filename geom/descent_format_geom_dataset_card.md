---
pretty_name: Meta-OMol25 Descent Formatted GEOM v1.0
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

# Dataset Card for Meta-OMol25 Descent Formatted GEOM v1.0

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

- Meta-OMol25 rows: 8953866
- Metadata rows: 8327720
- Smee rows: 8327720
- Failed rows: 626146
- Failed unique indices: 626146
- Accounted total (metadata + failed - overlap): 8953866
- All Meta-OMol25 structures accounted for: True
- Uncovered Meta-OMol25 rows: 0

### Metadata Split and Chemistry Summary

- Sample size used for summary: 8327720
- Sample split counts: {
  "val": 5591,
  "train": 8322129
}
- Sample charge range: [-4, 3]
- Sample molecular weight (g/mol, min / max / mean): 16.043000 / 1550.190820 / 385.282132
- Molecular weight sample count: 8327720

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

### Smee Schema

- smiles: Value('string')
- coords: List(Value('float32'))
- energy: List(Value('float32'))
- forces: List(Value('float32'))

Sample energy stats (kcal/mol):
- Min: -7202909.000000
- Max: -25408.492188
- Mean: -975299.030531

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
- ValueError('Inconsistent charge with target!'): 41075
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry S.'): 16764
- InconsistentStereochemistryError('Programming error: OpenEye atom stereochemistry assumptions failed. The atom in the oemol has stereochemistry None and the atom in the offmol has stereochemistry R.'): 14028
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry Z.'): 158
- InconsistentStereochemistryError('Programming error: OpenEye bond stereochemistry assumptions failed. The bond in the oemol has stereochemistry None and the bond in the offmol has stereochemistry E.'): 83
- AtomValenceException('Explicit valence for atom # 11 N, 4, is greater than permitted'): 40
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [Cl-].[H]c1c([H])c2c(c([H])c1OC(=O)N(C([H])([H])[C-]([H])[H])C([H])([H])C([H])([H])Cl)C([H])([H])C([H])([H])[C@@]1([H])[C@]3([H])C([H])([H])C([H])([H])[C@]([H])(OP([O])([O])=O)[C@@]3(C([H])([H])[H])C([H])([H])C([H])([H])[C@]21[H].'): 36
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H].[H]C(OC(=O)C1=NN(C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H])C(=O)c2c1c([H])c([H])c([H])c2[H])=C([O])C1=C(N([H])[H])N(C([H])([H])[H])C(=O)N(C([H])([H])[H])C1=O.'): 35
- AtomValenceException('Explicit valence for atom # 25 N, 4, is greater than permitted'): 32
- RadicalsNotSupportedError('The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. Found 1 radical electrons on molecule [H].[H][C](C([H])([H])c1c(OC([H])([H])[H])c([H])c(SC([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H])c(OC([H])([H])[H])c1[H])[N+]([H])(C([H])([H])[H])C([H])([H])[H].'): 29

## Dataset Card Authors

- Jennifer A Clark (Open Force Field Initiative); jaclark5

## Dataset Card Contact

- Primary contact: info@openforcefield.org
