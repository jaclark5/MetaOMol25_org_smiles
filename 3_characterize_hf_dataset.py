import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Optional

from ase.data import atomic_masses, chemical_symbols
from ase.db import connect
from datasets import Dataset, load_from_disk

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_HF_ROOT = "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets"
DEFAULT_ASE_ROOT = "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/ase_databases"

ATOMIC_MASSES = {
    chemical_symbols[z]: float(atomic_masses[z])
    for z in range(1, len(chemical_symbols))
    if chemical_symbols[z]
}


@dataclass
class PathResolution:
    dataset_root: str
    metadata_path: str
    smee_path: str
    failed_rows_path: str
    layout: str


def resolve_dataset_paths(hf_root: str, dataset: str) -> PathResolution:
    nested_root = os.path.join(hf_root, dataset)
    nested = {
        "dataset_root": nested_root,
        "metadata": os.path.join(nested_root, f"{dataset}_metadata"),
        "smee": os.path.join(nested_root, f"{dataset}_smee"),
        "failed": os.path.join(nested_root, f"{dataset}_failed_rows.jsonl"),
    }
    flat = {
        "dataset_root": hf_root,
        "metadata": os.path.join(hf_root, f"{dataset}_metadata"),
        "smee": os.path.join(hf_root, f"{dataset}_smee"),
        "failed": os.path.join(hf_root, f"{dataset}_failed_rows.jsonl"),
    }

    nested_exists = all(os.path.exists(nested[key]) for key in ["metadata", "smee", "failed"])
    flat_exists = all(os.path.exists(flat[key]) for key in ["metadata", "smee", "failed"])

    if nested_exists:
        return PathResolution(
            dataset_root=nested["dataset_root"],
            metadata_path=nested["metadata"],
            smee_path=nested["smee"],
            failed_rows_path=nested["failed"],
            layout="nested",
        )

    if flat_exists:
        return PathResolution(
            dataset_root=flat["dataset_root"],
            metadata_path=flat["metadata"],
            smee_path=flat["smee"],
            failed_rows_path=flat["failed"],
            layout="flat",
        )

    missing = {
        "nested": {k: v for k, v in nested.items() if k in ["metadata", "smee", "failed"] and not os.path.exists(v)},
        "flat": {k: v for k, v in flat.items() if k in ["metadata", "smee", "failed"] and not os.path.exists(v)},
    }
    raise FileNotFoundError(
        "Could not resolve dataset paths. Missing files: " + json.dumps(missing, indent=2)
    )


def safe_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple)):
        for entry in value:
            out = safe_numeric(entry)
            if out is not None:
                return out
        return None
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        return None
    return None


def extract_atomic_symbols(row: dict[str, Any]) -> list[str]:
    for key in ("atomic_symbols", "symbols", "elements"):
        value = row.get(key)
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return [str(v) for v in value]
    return []


def compute_molecular_weight(symbols: list[str]) -> Optional[float]:
    if not symbols:
        return None
    total = 0.0
    for symbol in symbols:
        mass = ATOMIC_MASSES.get(symbol)
        if mass is None:
            return None
        total += mass
    return total


def molecular_weight_from_formula(formula: Any) -> Optional[float]:
    if not isinstance(formula, str):
        return None
    text = formula.strip()
    if not text:
        return None

    # Simple chemical formula parser (e.g. C6H6O). Rejects unsupported syntax.
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not tokens:
        return None

    consumed = "".join(symbol + count for symbol, count in tokens)
    if consumed != text:
        return None

    total = 0.0
    for symbol, count_text in tokens:
        mass = ATOMIC_MASSES.get(symbol)
        if mass is None:
            return None
        count = int(count_text) if count_text else 1
        total += mass * count
    return total


def sampled_indices(n_rows: int, sample_size: int) -> list[int]:
    if n_rows <= 0:
        return []
    if sample_size == -1 or (sample_size > 0 and n_rows <= sample_size):
        return list(range(n_rows))

    step = n_rows / float(sample_size)
    indices = []
    for i in range(sample_size):
        idx = int(i * step)
        if idx >= n_rows:
            idx = n_rows - 1
        indices.append(idx)

    # Preserve order while de-duplicating boundary collisions.
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def iter_dataset_rows(ds: Dataset, indices: Optional[list[int]] = None) -> Iterable[dict[str, Any]]:
    if indices is None:
        for i in range(len(ds)):
            yield ds[i]
    else:
        for i in indices:
            yield ds[i]


def progress_wrap(iterable: Iterable[Any], total: Optional[int], desc: str, enabled: bool) -> Iterable[Any]:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def summarize_common_dataset(ds: Dataset, sample_size: int, show_progress: bool = True) -> dict[str, Any]:
    rows = len(ds)
    features = {k: str(v) for k, v in ds.features.items()} if ds.features is not None else {}
    sample_idx = sampled_indices(rows, sample_size)

    null_counts = Counter()
    rows_iter = progress_wrap(
        iter_dataset_rows(ds, sample_idx),
        total=len(sample_idx),
        desc="Stats: common",
        enabled=show_progress,
    )
    for row in rows_iter:
        for key, value in row.items():
            if value is None:
                null_counts[key] += 1

    return {
        "rows": rows,
        "columns": ds.column_names,
        "features": features,
        "sample_size_used": len(sample_idx),
        "sample_null_counts": dict(null_counts),
    }


def summarize_metadata_dataset(ds: Dataset, sample_size: int, show_progress: bool = True) -> dict[str, Any]:
    out = summarize_common_dataset(ds, sample_size, show_progress=show_progress)
    sample_idx = sampled_indices(len(ds), sample_size)

    split_counts = Counter()
    charge_counts = Counter()
    flag_elements = Counter()
    flag_charge = Counter()
    flag_spin = Counter()
    molecular_weight_values = []

    rows_iter = progress_wrap(
        iter_dataset_rows(ds, sample_idx),
        total=len(sample_idx),
        desc="Stats: metadata",
        enabled=show_progress,
    )
    for row in rows_iter:
        split = row.get("OMol25_split")
        if split is not None:
            split_counts[str(split)] += 1

        charge = row.get("charge")
        if charge is not None:
            charge_counts[str(charge)] += 1

        if "OpenFF_Elements" in row:
            flag_elements[str(bool(row.get("OpenFF_Elements")))] += 1
        if "OpenFF_abs(q)<=1" in row:
            flag_charge[str(bool(row.get("OpenFF_abs(q)<=1")))] += 1
        if "OpenFF_spin=1" in row:
            flag_spin[str(bool(row.get("OpenFF_spin=1")))] += 1

        mw = molecular_weight_from_formula(row.get("formula"))
        if mw is not None:
            molecular_weight_values.append(mw)

    molecular_weight_stats = {}
    if molecular_weight_values:
        molecular_weight_stats = {
            "min": min(molecular_weight_values),
            "max": max(molecular_weight_values),
            "mean": sum(molecular_weight_values) / len(molecular_weight_values),
            "n": len(molecular_weight_values),
        }

    out.update(
        {
            "sample_split_counts": dict(split_counts),
            "sample_charge_counts": dict(charge_counts),
            "sample_openff_elements_flag": dict(flag_elements),
            "sample_openff_abs_q_le_1_flag": dict(flag_charge),
            "sample_openff_spin_1_flag": dict(flag_spin),
            "sample_molecular_weight_stats": molecular_weight_stats,
        }
    )
    return out


def summarize_smee_dataset(ds: Dataset, sample_size: int, show_progress: bool = True) -> dict[str, Any]:
    out = summarize_common_dataset(ds, sample_size, show_progress=show_progress)
    sample_idx = sampled_indices(len(ds), sample_size)

    n_atoms_counts = Counter()
    coord_dims = Counter()
    force_dims = Counter()
    energy_values = []
    molecular_weight_values = []
    molecular_weight_source_counts = Counter()

    rows_iter = progress_wrap(
        iter_dataset_rows(ds, sample_idx),
        total=len(sample_idx),
        desc="Stats: smee",
        enabled=show_progress,
    )
    for row in rows_iter:
        coords = row.get("coords")
        forces = row.get("forces")
        energy = safe_numeric(row.get("energy"))
        coords_list = coords if isinstance(coords, list) else None
        forces_list = forces if isinstance(forces, list) else None

        if coords_list is not None:
            if coords_list and isinstance(coords_list[0], list):
                n_atoms_counts[len(coords_list)] += 1
                coord_dims[len(coords_list[0])] += 1
            elif len(coords_list) % 3 == 0:
                n_atoms_counts[len(coords_list) // 3] += 1
                coord_dims[3] += 1

        if forces_list is not None:
            if forces_list and isinstance(forces_list[0], list):
                force_dims[len(forces_list[0])] += 1
            elif len(forces_list) % 3 == 0:
                force_dims[3] += 1
            else:
                force_dims[-1] += 1

        if energy is not None:
            energy_values.append(energy)

        symbols = []
        if isinstance(row.get("atomic_symbols"), list):
            symbols = extract_atomic_symbols({"atomic_symbols": row.get("atomic_symbols")})
            if symbols:
                molecular_weight_source_counts["atomic_symbols"] += 1
        elif isinstance(row.get("symbols"), list):
            symbols = extract_atomic_symbols({"symbols": row.get("symbols")})
            if symbols:
                molecular_weight_source_counts["symbols"] += 1
        elif isinstance(row.get("elements"), list):
            symbols = extract_atomic_symbols({"elements": row.get("elements")})
            if symbols:
                molecular_weight_source_counts["elements"] += 1

        mw = compute_molecular_weight(symbols)
        if mw is not None:
            molecular_weight_values.append(mw)

    energy_stats = {}
    if energy_values:
        energy_stats = {
            "min": min(energy_values),
            "max": max(energy_values),
            "mean": sum(energy_values) / len(energy_values),
        }

    molecular_weight_stats = {}
    if molecular_weight_values:
        molecular_weight_stats = {
            "min": min(molecular_weight_values),
            "max": max(molecular_weight_values),
            "mean": sum(molecular_weight_values) / len(molecular_weight_values),
            "n": len(molecular_weight_values),
        }

    out.update(
        {
            "sample_coord_dim_distribution": dict(coord_dims),
            "sample_force_dim_distribution": dict(force_dims),
            "sample_energy_stats": energy_stats,
            "sample_molecular_weight_stats": molecular_weight_stats,
            "sample_molecular_weight_source_counts": dict(molecular_weight_source_counts),
        }
    )
    return out


def load_failed_rows(path: str) -> dict[str, Any]:
    failed_rows = []
    parse_errors = 0
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                failed_rows.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1

    index_counter = Counter()
    error_counter = Counter()

    for row in failed_rows:
        idx = row.get("index")
        if isinstance(idx, int):
            index_counter[idx] += 1
        error_counter[str(row.get("error", "<missing error>"))] += 1

    return {
        "rows_total": len(failed_rows),
        "json_parse_errors": parse_errors,
        "unique_failed_indices": len(index_counter),
        "duplicate_failed_index_rows": sum(v - 1 for v in index_counter.values() if v > 1),
        "top_errors": error_counter.most_common(20),
        "index_counter": index_counter,
    }


def index_coverage_check(
    metadata_ds: Dataset,
    failed_index_counter: Counter,
    ase_total: int,
    strict_index_max: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    strict = strict_index_max == -1 or ase_total <= strict_index_max
    seen = bytearray(ase_total) if strict and ase_total > 0 else None

    metadata_rows = len(metadata_ds)
    metadata_with_openff_id = 0
    metadata_without_openff_id = 0
    metadata_out_of_range = 0
    metadata_non_int = 0
    metadata_duplicate_indices = 0

    coverage_meta_unique = 0
    coverage_failed_unique = 0
    overlap_meta_failed = 0

    metadata_iter = progress_wrap(
        iter_dataset_rows(metadata_ds),
        total=len(metadata_ds),
        desc="ASE compare: metadata",
        enabled=show_progress,
    )
    for row in metadata_iter:
        idx = row.get("OpenFF_id")
        if idx is None:
            metadata_without_openff_id += 1
            continue
        metadata_with_openff_id += 1

        if not isinstance(idx, int):
            metadata_non_int += 1
            continue

        if idx < 0 or idx >= ase_total:
            metadata_out_of_range += 1
            continue

        if seen is not None:
            if seen[idx] == 0:
                seen[idx] = 1
                coverage_meta_unique += 1
            elif seen[idx] == 1:
                metadata_duplicate_indices += 1
            elif seen[idx] == 2:
                overlap_meta_failed += 1
                metadata_duplicate_indices += 1
            elif seen[idx] == 3:
                metadata_duplicate_indices += 1
        else:
            # In non-strict mode, treat rows as unique by count only.
            coverage_meta_unique += 1

    failed_out_of_range = 0
    failed_non_int = 0

    failed_iter = progress_wrap(
        failed_index_counter.items(),
        total=len(failed_index_counter),
        desc="ASE compare: failed",
        enabled=show_progress,
    )
    for idx, count in failed_iter:
        if not isinstance(idx, int):
            failed_non_int += count
            continue
        if idx < 0 or idx >= ase_total:
            failed_out_of_range += count
            continue

        if seen is not None:
            if seen[idx] == 0:
                seen[idx] = 2
                coverage_failed_unique += 1
            elif seen[idx] == 1:
                seen[idx] = 3
                overlap_meta_failed += 1
                coverage_failed_unique += 1
            elif seen[idx] in (2, 3):
                # Unique index counter already deduplicates this key.
                pass
        else:
            coverage_failed_unique += 1

    uncovered_indices = None
    uncovered_count = None
    if seen is not None:
        uncovered_count = sum(1 for v in seen if v == 0)
        if uncovered_count and uncovered_count <= 1000:
            uncovered_indices = [i for i, v in enumerate(seen) if v == 0]
    else:
        # Non-strict mode: best-effort estimate from counts only.
        uncovered_count = max(ase_total - (coverage_meta_unique + coverage_failed_unique), 0)

    accounted_total = coverage_meta_unique + coverage_failed_unique - overlap_meta_failed

    all_accounted_for = (
        metadata_without_openff_id == 0
        and metadata_non_int == 0
        and metadata_out_of_range == 0
        and failed_non_int == 0
        and failed_out_of_range == 0
        and overlap_meta_failed == 0
        and metadata_duplicate_indices == 0
        and accounted_total == ase_total
        and (uncovered_count == 0)
    )

    return {
        "strict_mode": strict,
        "strict_index_max": strict_index_max,
        "ase_total_rows": ase_total,
        "metadata_rows": metadata_rows,
        "metadata_rows_with_openff_id": metadata_with_openff_id,
        "metadata_rows_without_openff_id": metadata_without_openff_id,
        "metadata_non_int_openff_id_rows": metadata_non_int,
        "metadata_out_of_range_openff_id_rows": metadata_out_of_range,
        "metadata_duplicate_indices": metadata_duplicate_indices,
        "failed_unique_indices": len(failed_index_counter),
        "failed_non_int_indices": failed_non_int,
        "failed_out_of_range_indices": failed_out_of_range,
        "coverage_meta_unique": coverage_meta_unique,
        "coverage_failed_unique": coverage_failed_unique,
        "overlap_meta_failed": overlap_meta_failed,
        "accounted_total_after_overlap_adjustment": accounted_total,
        "uncovered_count": uncovered_count,
        "uncovered_indices_if_small": uncovered_indices,
        "all_ase_structures_accounted_for": all_accounted_for,
    }


def compare_metadata_smee_alignment(metadata_ds: Dataset, smee_ds: Dataset, sample_size: int) -> dict[str, Any]:
    len_meta = len(metadata_ds)
    len_smee = len(smee_ds)
    equal_rows = len_meta == len_smee

    indices = sampled_indices(min(len_meta, len_smee), sample_size)
    smiles_mismatch = 0

    for idx in indices:
        meta_smiles = metadata_ds[idx].get("smiles")
        smee_smiles = smee_ds[idx].get("smiles")
        if meta_smiles != smee_smiles:
            smiles_mismatch += 1

    return {
        "metadata_rows": len_meta,
        "smee_rows": len_smee,
        "row_count_match": equal_rows,
        "sample_size_used": len(indices),
        "sample_smiles_mismatch_count": smiles_mismatch,
    }


def validate_descent_smee_processability(smee_ds: Dataset, max_rows: int) -> dict[str, Any]:
    result = {
        "smee_importable": False,
        "descent_importable": False,
        "descent_create_dataset_ok": False,
        "validation_rows_used": 0,
        "error": None,
    }

    try:
        import smee  # noqa: F401

        result["smee_importable"] = True
    except Exception as exc:
        result["error"] = f"smee import failed: {exc!r}"

    try:
        import descent.targets.energy as energy_target

        result["descent_importable"] = True
    except Exception as exc:
        previous = result["error"]
        new_error = f"descent import failed: {exc!r}"
        result["error"] = f"{previous} | {new_error}" if previous else new_error
        return result

    n_validate = min(len(smee_ds), max_rows)
    sample_rows = []
    for i in range(n_validate):
        row = smee_ds[i]
        sample_rows.append(
            {
                "smiles": row.get("smiles"),
                "coords": row.get("coords"),
                "energy": row.get("energy"),
                "forces": row.get("forces"),
            }
        )

    result["validation_rows_used"] = n_validate

    try:
        rebuilt = energy_target.create_dataset(sample_rows)
        result["descent_create_dataset_ok"] = len(rebuilt) == n_validate
    except Exception as exc:
        previous = result["error"]
        new_error = f"descent create_dataset failed: {exc!r}"
        result["error"] = f"{previous} | {new_error}" if previous else new_error

    return result


def build_text_report(report: dict[str, Any]) -> str:
    lines = []

    lines.append(f"Dataset: {report['dataset']}")
    lines.append(f"Layout detected: {report['paths']['layout']}")
    lines.append(f"Metadata path: {report['paths']['metadata_path']}")
    lines.append(f"Smee path: {report['paths']['smee_path']}")
    lines.append(f"Failed rows path: {report['paths']['failed_rows_path']}")
    lines.append("")

    coverage = report["coverage"]
    lines.append("Coverage check against ASE DB")
    lines.append(f"- ASE rows: {coverage['ase_total_rows']}")
    lines.append(f"- Metadata rows: {coverage['metadata_rows']}")
    lines.append(f"- Failed unique indices: {coverage['failed_unique_indices']}")
    lines.append(f"- Accounted total: {coverage['accounted_total_after_overlap_adjustment']}")
    lines.append(f"- Uncovered count: {coverage['uncovered_count']}")
    lines.append(f"- All accounted for: {coverage['all_ase_structures_accounted_for']}")
    lines.append("")

    alignment = report["alignment"]
    lines.append("Metadata/Smee alignment")
    lines.append(f"- Row count match: {alignment['row_count_match']}")
    lines.append(f"- Sample smiles mismatches: {alignment['sample_smiles_mismatch_count']}")
    lines.append("")

    processability = report["processability"]
    lines.append("smee/descent processability")
    lines.append(f"- smee importable: {processability['smee_importable']}")
    lines.append(f"- descent importable: {processability['descent_importable']}")
    lines.append(f"- descent.create_dataset(sample) ok: {processability['descent_create_dataset_ok']}")
    if processability.get("error"):
        lines.append(f"- validation error: {processability['error']}")
    lines.append("")

    lines.append("Metadata structure")
    lines.append(f"- Columns: {', '.join(report['metadata']['columns'])}")
    lines.append(f"- Rows: {report['metadata']['rows']}")
    lines.append("")

    lines.append("Smee structure")
    lines.append(f"- Columns: {', '.join(report['smee']['columns'])}")
    lines.append(f"- Rows: {report['smee']['rows']}")

    return "\n".join(lines)


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def top_items_to_text(items: Any, n_max: int = 10) -> str:
    if not isinstance(items, list) or not items:
        return ""
    lines = []
    for item in items[:n_max]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lines.append(f"- {item[0]}: {item[1]}")
    return "\n".join(lines)


def list_to_bullets(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    return "\n".join(f"- {str(item)}" for item in items)


def schema_to_typed_bullets(columns: Any, features: Any) -> str:
    if not isinstance(columns, list) or not columns:
        return ""
    feature_map = features if isinstance(features, dict) else {}
    lines = []
    for column in columns:
        dtype = str(feature_map.get(str(column), "unknown"))
        lines.append(f"- {column}: {dtype}")
    return "\n".join(lines)


def build_card_tokens(report: dict[str, Any]) -> dict[str, str]:
    metadata = report["metadata"]
    smee = report["smee"]
    coverage = report["coverage"]
    alignment = report["alignment"]
    failed = report["failed_rows"]
    processability = report["processability"]

    sample_charge_counts = metadata.get("sample_charge_counts", {})
    sample_charge_keys = []
    for k in sample_charge_counts.keys():
        try:
            sample_charge_keys.append(float(k))
        except Exception:
            pass

    charge_min = min(sample_charge_keys) if sample_charge_keys else None
    charge_max = max(sample_charge_keys) if sample_charge_keys else None

    energy_stats = smee.get("sample_energy_stats", {})
    mw_stats = metadata.get("sample_molecular_weight_stats", {}) or smee.get("sample_molecular_weight_stats", {})

    def value_or_na(text: str) -> str:
        return text if text else "N/A"

    tokens = {
        "{{PRETTY_NAME}}": str(report.get("card_pretty_name", report.get("dataset", ""))),
        "{{DATASET_NAME}}": str(report.get("dataset", "")),
        "{{RUN_DATE}}": str(date.today()),
        "{{LICENSE}}": "CC-BY-4.0",
        "{{CURATED_BY}}": "Jennifer A Clark; jaclark5",
        "{{FUNDED_BY}}": "Open Force Field Initiative",
        "{{SHARED_BY}}": "Open Force Field Initiative, Open Molecular Software Foundation",
        "{{DATASET_VERSION}}": "v1.0",
        "{{CONTACT_EMAIL}}": "info@openforcefield.org",
        "{{REPOSITORY_URL}}": "https://huggingface.co/facebook/OMol25",
        "{{DATASET_DESCRIPTION}}": (
            "Meta-OMol25 provides molecular structures, coordinates, energies, and forces, and we derived mapped SMILES "
            "for broad OpenFF parameter fitting workflows. This release is designed for general fitting and "
            "evaluation of van der Waals and valence terms."
        ),
        "{{TOTAL_ASE_ROWS}}": str(coverage.get("ase_total_rows", "")),
        "{{TOTAL_METADATA_ROWS}}": str(metadata.get("rows", "")),
        "{{TOTAL_SMEE_ROWS}}": str(smee.get("rows", "")),
        "{{TOTAL_FAILED_ROWS}}": str(failed.get("rows_total", "")),
        "{{TOTAL_FAILED_UNIQUE_INDICES}}": str(failed.get("unique_failed_indices", "")),
        "{{ACCOUNTED_TOTAL}}": str(coverage.get("accounted_total_after_overlap_adjustment", "")),
        "{{ALL_ACCOUNTED_FOR}}": str(coverage.get("all_ase_structures_accounted_for", "")),
        "{{UNCOVERED_COUNT}}": str(coverage.get("uncovered_count", "")),
        "{{ROW_COUNT_MATCH}}": str(alignment.get("row_count_match", "")),
        "{{SMILES_MISMATCH_COUNT}}": str(alignment.get("sample_smiles_mismatch_count", "")),
        "{{SAMPLE_SIZE_USED}}": str(metadata.get("sample_size_used", "")),
        "{{SPLIT_COUNTS}}": json.dumps(metadata.get("sample_split_counts", {}), indent=2),
        "{{CHARGE_RANGE_SAMPLE}}": (
            f"[{format_float(charge_min, 0)}, {format_float(charge_max, 0)}]" if sample_charge_keys else "N/A"
        ),
        "{{METADATA_COLUMNS_BULLETS}}": schema_to_typed_bullets(
            metadata.get("columns", []), metadata.get("features", {})
        ),
        "{{SMEE_COLUMNS_BULLETS}}": schema_to_typed_bullets(
            smee.get("columns", []), smee.get("features", {})
        ),
        "{{ENERGY_MIN_SAMPLE}}": value_or_na(format_float(energy_stats.get("min"), 6)),
        "{{ENERGY_MAX_SAMPLE}}": value_or_na(format_float(energy_stats.get("max"), 6)),
        "{{ENERGY_MEAN_SAMPLE}}": value_or_na(format_float(energy_stats.get("mean"), 6)),
        "{{MOLECULAR_WEIGHT_MIN_SAMPLE}}": value_or_na(format_float(mw_stats.get("min"), 6)),
        "{{MOLECULAR_WEIGHT_MAX_SAMPLE}}": value_or_na(format_float(mw_stats.get("max"), 6)),
        "{{MOLECULAR_WEIGHT_MEAN_SAMPLE}}": value_or_na(format_float(mw_stats.get("mean"), 6)),
        "{{MOLECULAR_WEIGHT_N_SAMPLE}}": value_or_na(str(mw_stats.get("n", ""))),
        "{{PROCESSABILITY_SMEE_IMPORTABLE}}": str(processability.get("smee_importable", "")),
        "{{PROCESSABILITY_DESCENT_IMPORTABLE}}": str(processability.get("descent_importable", "")),
        "{{PROCESSABILITY_CREATE_DATASET_OK}}": str(processability.get("descent_create_dataset_ok", "")),
        "{{PROCESSABILITY_VALIDATION_ROWS}}": str(processability.get("validation_rows_used", "")),
        "{{PROCESSABILITY_ERROR}}": str(processability.get("error") or ""),
        "{{ASE_DB_PATH}}": report.get("ase_db_path", ""),
        "{{HF_METADATA_PATH}}": report.get("paths", {}).get("metadata_path", ""),
        "{{HF_SMEE_PATH}}": report.get("paths", {}).get("smee_path", ""),
        "{{FAILED_ROWS_PATH}}": report.get("paths", {}).get("failed_rows_path", ""),
        "{{TOP_FAILURE_MODES}}": top_items_to_text(failed.get("top_errors", []), n_max=10),
        "{{DATASET_CARD_AUTHORS}}": "Jennifer A Clark (Open Force Field Initiative); jaclark5",
    }
    return tokens


def render_template(template: str, tokens: dict[str, str]) -> str:
    text = template
    for key, value in tokens.items():
        text = text.replace(key, value)
    return text


def default_hf_card_template() -> str:
    return """---
pretty_name: {{PRETTY_NAME}}
license: {{LICENSE}}
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

# Dataset Card for {{PRETTY_NAME}}

## Dataset Details

### Dataset Description

{{DATASET_DESCRIPTION}}

- Curated by: {{CURATED_BY}}
- Funded by: {{FUNDED_BY}}
- Shared by: {{SHARED_BY}}
- License: {{LICENSE}}
- Dataset version: {{DATASET_VERSION}}

### Dataset Sources

- Repository: {{REPOSITORY_URL}}
- Hugging Face dataset repository: {{HF_REPO_URL}}

## Uses

### Direct Use

- Fit or benchmark van der Waals parameters.
- Fit or benchmark valence terms (bonds, angles, torsions).
- Provide aligned molecular metadata and per-structure coordinates / energies / forces for OpenFF workflows.

## Dataset Structure

### Overall Statistics

- Meta-OMol25 rows: {{TOTAL_ASE_ROWS}}
- Metadata rows: {{TOTAL_METADATA_ROWS}}
- Smee rows: {{TOTAL_SMEE_ROWS}}
- Failed rows: {{TOTAL_FAILED_ROWS}}
- Failed unique indices: {{TOTAL_FAILED_UNIQUE_INDICES}}
- Accounted total (metadata + failed - overlap): {{ACCOUNTED_TOTAL}}
- All Meta-OMol25 structures accounted for: {{ALL_ACCOUNTED_FOR}}
- Uncovered Meta-OMol25 rows: {{UNCOVERED_COUNT}}

### Metadata Split and Chemistry Summary

- Sample size used for summary: {{SAMPLE_SIZE_USED}}
- Sample split counts: {{SPLIT_COUNTS}}
- Sample charge range: {{CHARGE_RANGE_SAMPLE}}
- Sample molecular weight (g/mol, min / max / mean): {{MOLECULAR_WEIGHT_MIN_SAMPLE}} / {{MOLECULAR_WEIGHT_MAX_SAMPLE}} / {{MOLECULAR_WEIGHT_MEAN_SAMPLE}}
- Molecular weight sample count: {{MOLECULAR_WEIGHT_N_SAMPLE}}

### Metadata Schema

{{METADATA_COLUMNS_BULLETS}}

### Smee Schema

{{SMEE_COLUMNS_BULLETS}}

Sample energy stats (kcal/mol):
- Min: {{ENERGY_MIN_SAMPLE}}
- Max: {{ENERGY_MAX_SAMPLE}}
- Mean: {{ENERGY_MEAN_SAMPLE}}

### Cross-Table Consistency

- Metadata and smee row count match: {{ROW_COUNT_MATCH}}
- Sample metadata/smee SMILES mismatches: {{SMILES_MISMATCH_COUNT}}

## Dataset Creation

### Curation Rationale

- This dataset is intended for broad force-field parameterization workloads and diagnostics.
- This release emphasizes direct support for van der Waals and valence fitting tasks.

### Data Collection and Processing

- Upstream rows are converted into metadata and smee datasets.
- Rows that fail conversion are logged in failed_rows JSONL and used for reconciliation.
- Coverage checks validate whether all Meta-OMol25 rows are represented by either successful records or failed-row entries.

### Quality and Validation

- smee importable: {{PROCESSABILITY_SMEE_IMPORTABLE}}
- descent importable: {{PROCESSABILITY_DESCENT_IMPORTABLE}}
- descent create_dataset on sample succeeded: {{PROCESSABILITY_CREATE_DATASET_OK}}
- Validation sample size: {{PROCESSABILITY_VALIDATION_ROWS}}
- Validation error (if any): {{PROCESSABILITY_ERROR}}

Top failure modes:
{{TOP_FAILURE_MODES}}

## Dataset Card Authors

- {{DATASET_CARD_AUTHORS}}

## Dataset Card Contact

- Primary contact: {{CONTACT_EMAIL}}
"""


def write_hf_dataset_card(report: dict[str, Any], template_path: Optional[str], output_path: str) -> None:
    if template_path:
        with open(template_path, "r") as handle:
            template = handle.read()
    else:
        template = default_hf_card_template()

    tokens = build_card_tokens(report)
    card_text = render_template(template, tokens)

    with open(output_path, "w") as handle:
        handle.write(card_text.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize Meta-OMol25 Hugging Face datasets (metadata + smee), "
            "validate processability with descent/smee, and verify ASE coverage "
            "using failed_rows JSONL."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. geom_orca6")
    parser.add_argument("--hf-root", default=DEFAULT_HF_ROOT, help="Root containing Hugging Face datasets")
    parser.add_argument("--ase-root", default=DEFAULT_ASE_ROOT, help="Root containing ASE DB files")
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Full path to Hugging Face metadata dataset directory",
    )
    parser.add_argument(
        "--smee-path",
        default=None,
        help="Full path to Hugging Face smee dataset directory",
    )
    parser.add_argument(
        "--failed-rows-path",
        default=None,
        help="Full path to failed_rows JSONL",
    )
    parser.add_argument(
        "--ase-db-path",
        default=None,
        help="Full path to ASE database file",
    )
    parser.add_argument(
        "--card-pretty-name",
        default=None,
        help="Dataset-card pretty_name (can differ from --dataset)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Sample size used for descriptive statistics on metadata/smee; use -1 to analyze all rows",
    )
    parser.add_argument(
        "--processability-rows",
        type=int,
        default=256,
        help="Number of smee rows to test via descent.targets.energy.create_dataset",
    )
    parser.add_argument(
        "--strict-index-max",
        type=int,
        default=-1,
        help="Max ASE rows to use strict bytearray index reconciliation; use -1 to always use strict mode",
    )
    parser.add_argument("--output-json", default=None, help="Optional output JSON report path")
    parser.add_argument("--output-text", default=None, help="Optional output text report path")
    parser.add_argument(
        "--output-card",
        default=None,
        help="Optional output markdown dataset-card path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for default outputs (JSON report, text report, dataset card). "
            "Ignored for any file where an explicit --output-json/--output-text/--output-card is provided."
        ),
    )
    parser.add_argument(
        "--card-template",
        default=None,
        help="Optional markdown template path with {{TOKENS}} placeholders",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        help="Disable progress bars",
    )
    args = parser.parse_args()

    if args.sample_size <= 0 and args.sample_size != -1:
        raise ValueError("--sample-size must be a positive integer, or -1 to analyze all rows")
    if args.strict_index_max <= 0 and args.strict_index_max != -1:
        raise ValueError("--strict-index-max must be a positive integer, or -1 to always use strict mode")

    show_progress = not args.no_progress

    dataset = args.dataset
    explicit_paths = [args.metadata_path, args.smee_path, args.failed_rows_path, args.ase_db_path]
    using_explicit_paths = any(value is not None for value in explicit_paths)

    if using_explicit_paths:
        missing = []
        if args.metadata_path is None:
            missing.append("--metadata-path")
        if args.smee_path is None:
            missing.append("--smee-path")
        if args.failed_rows_path is None:
            missing.append("--failed-rows-path")
        if args.ase_db_path is None:
            missing.append("--ase-db-path")
        if missing:
            raise ValueError(
                "When using explicit full paths, provide all of: " + ", ".join(missing)
            )

        for required_path, label in [
            (args.metadata_path, "metadata dataset"),
            (args.smee_path, "smee dataset"),
            (args.failed_rows_path, "failed rows JSONL"),
            (args.ase_db_path, "ASE database"),
        ]:
            if not os.path.exists(required_path):
                raise FileNotFoundError(f"{label} not found: {required_path}")

        dataset_root = os.path.dirname(os.path.abspath(args.metadata_path))
        paths = PathResolution(
            dataset_root=dataset_root,
            metadata_path=args.metadata_path,
            smee_path=args.smee_path,
            failed_rows_path=args.failed_rows_path,
            layout="explicit",
        )
        ase_db_path = args.ase_db_path
    else:
        paths = resolve_dataset_paths(args.hf_root, dataset)
        ase_db_path = os.path.join(args.ase_root, f"{dataset}.db")

        if not os.path.exists(ase_db_path):
            raise FileNotFoundError(f"ASE database not found: {ase_db_path}")

    metadata_ds = load_from_disk(paths.metadata_path)
    smee_ds = load_from_disk(paths.smee_path)

    failed_info = load_failed_rows(paths.failed_rows_path)
    ase_total = connect(ase_db_path).count()

    metadata_summary = summarize_metadata_dataset(metadata_ds, args.sample_size, show_progress=show_progress)
    smee_summary = summarize_smee_dataset(smee_ds, args.sample_size, show_progress=show_progress)
    alignment = compare_metadata_smee_alignment(metadata_ds, smee_ds, args.sample_size)

    coverage = index_coverage_check(
        metadata_ds=metadata_ds,
        failed_index_counter=failed_info["index_counter"],
        ase_total=ase_total,
        strict_index_max=args.strict_index_max,
        show_progress=show_progress,
    )

    processability = validate_descent_smee_processability(smee_ds, args.processability_rows)

    failed_summary = {k: v for k, v in failed_info.items() if k != "index_counter"}

    report: dict[str, Any] = {
        "dataset": dataset,
        "card_pretty_name": args.card_pretty_name or dataset,
        "paths": asdict(paths),
        "ase_db_path": ase_db_path,
        "metadata": metadata_summary,
        "smee": smee_summary,
        "failed_rows": failed_summary,
        "alignment": alignment,
        "coverage": coverage,
        "processability": processability,
    }

    output_root = args.output_dir or paths.dataset_root
    os.makedirs(output_root, exist_ok=True)

    default_json = os.path.join(output_root, f"{dataset}_characterization.json")
    default_txt = os.path.join(output_root, f"{dataset}_characterization.txt")
    default_card = os.path.join(output_root, f"{dataset}_dataset_card.md")
    output_json = args.output_json or default_json
    output_text = args.output_text or default_txt
    output_card = args.output_card or default_card

    with open(output_json, "w") as handle:
        json.dump(report, handle, indent=2)

    text_report = build_text_report(report)
    with open(output_text, "w") as handle:
        handle.write(text_report + "\n")

    write_hf_dataset_card(
        report=report,
        template_path=args.card_template,
        output_path=output_card,
    )

    print(text_report)
    print("")
    print(f"Full JSON report written to: {output_json}")
    print(f"Text report written to: {output_text}")
    print(f"Draft dataset card written to: {output_card}")


if __name__ == "__main__":
    main()
