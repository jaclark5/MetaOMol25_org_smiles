"""
SMILES Quality Assessment Module

Analyzes SMILES strings in MetaOMol25 datasets for structural issues:
- Orphaned hydrogen atoms (isolated [H] fragments)
- Radicals (unpaired electrons)
- Multiple disconnected fragments
- Charge inconsistencies
- Valence violations
- Other structural anomalies

Saves classification results as CSV and summary JSON.
"""

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass
class SmilesQualityResult:
    """Classification result for a single SMILES string."""
    index: int
    smiles: str
    is_valid_rdkit: bool
    issues: List[str]  # List of detected issues
    classification: str  # "OK", "WARNING", "ERROR"
    num_fragments: int
    has_orphaned_h: bool
    has_radical: bool
    has_charge_imbalance: bool
    has_implicit_h: bool


def classify_smiles(smiles: str, index: int) -> SmilesQualityResult:
    """
    Analyze a SMILES string and classify it.
    
    Returns SmilesQualityResult with classification and issue list.
    """
    issues = []
    classification = "OK"
    
    # Try to parse with RDKit
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    
    if mol is None:
        return SmilesQualityResult(
            index=index,
            smiles=smiles,
            is_valid_rdkit=False,
            issues=["Cannot parse with RDKit"],
            classification="ERROR",
            num_fragments=0,
            has_orphaned_h=False,
            has_radical=False,
            has_charge_imbalance=False,
            has_implicit_h=False,
        )
    
    is_valid_rdkit = True
    
    # Check for disconnected fragments
    frag_list = Chem.GetMolFrags(mol, asMols=False)
    num_fragments = len(frag_list)
    
    if num_fragments > 1:
        issues.append(f"Multiple fragments ({num_fragments})")
        classification = "WARNING"
    
    # Check for orphaned hydrogen atoms: isolated [H] fragments
    has_orphaned_h = False
    for atom in mol.GetAtoms():
        # Check for hydrogen with no bonds (orphaned)
        if atom.GetSymbol() == "H" and atom.GetDegree() == 0:
            has_orphaned_h = True
            issues.append("Orphaned hydrogen ([H] with no bonds)")
            classification = "ERROR"
            break
    
    # Check for radicals (unpaired electrons)
    has_radical = False
    radical_count = 0
    for atom in mol.GetAtoms():
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            has_radical = True
            radical_count += num_radicals
    
    if has_radical:
        issues.append(f"Radical electrons ({radical_count})")
        classification = "ERROR"
    
    # Check for implicit hydrogens (all H should be explicit)
    has_implicit_h = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H" and atom.GetTotalNumHs() > 0:
            has_implicit_h = True
            issues.append(f"Implicit hydrogens on {atom.GetSymbol()}")
            classification = "ERROR"
            break
    
    # Check for charge imbalance (should ideally sum to 0 or ±1)
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    has_charge_imbalance = False
    if abs(total_charge) > 3:  # Threshold for "imbalanced"
        has_charge_imbalance = True
        issues.append(f"High total charge ({total_charge})")
        classification = "WARNING"
    
    # Check for valence violations
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        issues.append(f"Valence error: {type(e).__name__}")
        if classification != "ERROR":
            classification = "WARNING"
    
    # Check for unusual atom types or bonds
    unusual_atoms = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]:
            unusual_atoms.append(symbol)
    
    if unusual_atoms:
        unique_unusual = list(set(unusual_atoms))
        issues.append(f"Unusual elements: {', '.join(unique_unusual)}")
        if classification == "OK":
            classification = "WARNING"
    
    return SmilesQualityResult(
        index=index,
        smiles=smiles,
        is_valid_rdkit=is_valid_rdkit,
        issues=issues,
        classification=classification,
        num_fragments=num_fragments,
        has_orphaned_h=has_orphaned_h,
        has_radical=has_radical,
        has_charge_imbalance=has_charge_imbalance,
        has_implicit_h=has_implicit_h,
    )


def analyze_dataset(repo_id: str, max_rows: int = -1, output_dir: Optional[str] = None) -> dict[str, Any]:
    """
    Analyze SMILES in a dataset for quality issues.
    
    Args:
        repo_id: Full HuggingFace Hub repository ID (e.g., "openforcefield/descent-format-ani2x")
        max_rows: Maximum rows to analyze (-1 for all)
        output_dir: Directory to save CSV and summary (default: current dir)
    
    Returns:
        Dictionary with analysis results.
    """
    
    print(f"Repository: {repo_id}")
    
    # Load metadata dataset from HF Hub
    print(f"Loading metadata from Hub...")
    metadata_ds = load_dataset(repo_id, name="metadata", split="train")
    
    n_total = len(metadata_ds)
    n_analyze = n_total if max_rows <= 0 else min(n_total, max_rows)
    
    print(f"Analyzing {n_analyze}/{n_total} rows...")
    
    # Analyze each SMILES
    results: List[SmilesQualityResult] = []
    iterator = range(n_analyze)
    if tqdm:
        iterator = tqdm(iterator, total=n_analyze, desc="Analyzing SMILES")
    
    for i in iterator:
        row = metadata_ds[i]
        smiles = row.get("smiles")
        if smiles:
            result = classify_smiles(smiles, i)
            results.append(result)
    
    # Aggregate statistics
    classifications = Counter(r.classification for r in results)
    issues_counter = Counter()
    for r in results:
        for issue in r.issues:
            issues_counter[issue] += 1
    
    # Extract dataset name from repo_id for output file naming
    dataset_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    # Compute summary statistics
    summary = {
        "repo_id": repo_id,
        "date": str(date.today()),
        "rows_analyzed": len(results),
        "rows_total": n_total,
        "classification_counts": dict(classifications),
        "top_issues": sorted(issues_counter.items(), key=lambda x: x[1], reverse=True)[:20],
        "stats": {
            "num_valid_rdkit": sum(1 for r in results if r.is_valid_rdkit),
            "num_orphaned_h": sum(1 for r in results if r.has_orphaned_h),
            "num_radicals": sum(1 for r in results if r.has_radical),
            "num_charge_imbalance": sum(1 for r in results if r.has_charge_imbalance),
            "num_implicit_h": sum(1 for r in results if r.has_implicit_h),
            "num_multifragment": sum(1 for r in results if r.num_fragments > 1),
            "avg_fragments": sum(r.num_fragments for r in results) / len(results) if results else 0,
        },
    }
    
    # Save outputs
    if output_dir is None:
        output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{dataset_name}_smiles_quality.csv")
    json_path = os.path.join(output_dir, f"{dataset_name}_smiles_quality_summary.json")
    
    # Write CSV
    with open(csv_path, "w") as f:
        f.write("index,smiles,classification,issues,num_fragments,orphaned_h,radical,charge_imbalance,implicit_h\n")
        for r in results:
            issues_str = "|".join(r.issues) if r.issues else "none"
            f.write(f'{r.index},"{r.smiles}",{r.classification},"{issues_str}",{r.num_fragments},{r.has_orphaned_h},{r.has_radical},{r.has_charge_imbalance},{r.has_implicit_h}\n')
    
    print(f"Saved CSV: {csv_path}")
    
    # Write JSON summary
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary: {json_path}")
    
    # Print text report
    print("\n" + "=" * 60)
    print("SMILES QUALITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Rows analyzed: {summary['rows_analyzed']}/{summary['rows_total']}")
    print(f"\nClassification Counts:")
    for classification, count in sorted(classifications.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(results)
        print(f"  {classification}: {count} ({pct:.2f}%)")
    
    print(f"\nQuality Statistics:")
    print(f"  Valid RDKit: {summary['stats']['num_valid_rdkit']}/{len(results)}")
    print(f"  Orphaned H atoms: {summary['stats']['num_orphaned_h']}")
    print(f"  Implicit H present: {summary['stats']['num_implicit_h']}")
    print(f"  Radicals: {summary['stats']['num_radicals']}")
    print(f"  Charge imbalance: {summary['stats']['num_charge_imbalance']}")
    print(f"  Multi-fragment: {summary['stats']['num_multifragment']}")
    print(f"  Avg fragments: {summary['stats']['avg_fragments']:.2f}")
    
    print(f"\nTop Issues (by frequency):")
    for issue, count in summary["top_issues"][:10]:
        pct = 100.0 * count / len(results)
        print(f"  {count:6d} ({pct:5.2f}%) {issue}")
    
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SMILES quality in MetaOMol25 datasets from HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Full HuggingFace Hub repository ID (e.g., openforcefield/descent-format-ani2x)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Maximum rows to analyze (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save CSV and JSON outputs (default: current directory)",
    )
    
    args = parser.parse_args()
    
    analyze_dataset(
        repo_id=args.repo_id,
        max_rows=args.max_rows,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
