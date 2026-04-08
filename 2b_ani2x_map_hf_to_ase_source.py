import argparse
import os
import shutil
from typing import Any, Optional

from ase.db import connect
from datasets import load_from_disk

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def progress_wrap(iterable, total: Optional[int], desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def normalize_for_hf(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize_for_hf(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_ase_data_field(row: Any, field: str) -> Any:
    if row is None:
        return None

    if hasattr(row, "data") and isinstance(row.data, dict):
        return row.data.get(field)

    value = None
    if hasattr(row, "get"):
        data = row.get("data")
        if isinstance(data, dict):
            value = data.get(field)

    return value


def build_ase_index_map(ase_db_path: str, show_progress: bool) -> tuple[dict[int, dict[str, Any]], int, int]:
    db = connect(ase_db_path)
    total = db.count()
    mapping: dict[int, dict[str, Any]] = {}
    lowdin_charges_rows = 0

    rows_iter = progress_wrap(
        db.select(),
        total=total,
        desc="Indexing ASE rows",
        enabled=show_progress,
    )

    for row_index, row in enumerate(rows_iter):
        source_raw = read_ase_data_field(row, "source")
        source = "" if source_raw is None else str(source_raw)

        lowdin_charges = normalize_for_hf(read_ase_data_field(row, "lowdin_charges"))
        if lowdin_charges is not None:
            lowdin_charges_rows += 1

        mapping[row_index] = {
            "source": source,
            "lowdin_charges": lowdin_charges,
        }

    return mapping, total, lowdin_charges_rows


def add_ase_fields_to_dataset(
    hf_dataset_path: str,
    ase_index_map: dict[int, dict[str, Any]],
    show_progress: bool,
) -> tuple[int, int, bool, bool, bool]:
    ds = load_from_disk(hf_dataset_path)
    total_rows = len(ds)

    missing_ase_matches = 0
    source_values = []
    lowdin_charges_values = []

    rows_iter = progress_wrap(
        range(total_rows),
        total=total_rows,
        desc="Mapping HF rows",
        enabled=show_progress,
    )

    for hf_row_index in rows_iter:
        row = ds[hf_row_index]
        openff_id = row.get("OpenFF_id")

        ase_source = ""
        ase_lowdin_charges = None

        if isinstance(openff_id, int):
            match = ase_index_map.get(openff_id)
            if match is not None:
                ase_source = match.get("source", "")
                ase_lowdin_charges = match.get("lowdin_charges")
            else:
                missing_ase_matches += 1
        else:
            missing_ase_matches += 1

        source_values.append(ase_source)
        lowdin_charges_values.append(ase_lowdin_charges)

    source_added = False
    lowdin_charges_added = False
    source_skipped = False

    if "source" in ds.column_names:
        source_skipped = True
    else:
        ds = ds.add_column("source", source_values)
        source_added = True

    if "lowdin_charges" not in ds.column_names and any(v is not None for v in lowdin_charges_values):
        ds = ds.add_column("lowdin_charges", lowdin_charges_values)
        lowdin_charges_added = True

    if not source_added and not lowdin_charges_added:
        return total_rows, missing_ase_matches, source_added, lowdin_charges_added, source_skipped

    temp_output = hf_dataset_path + "__tmp_with_ase_fields"
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    ds.save_to_disk(temp_output)

    backup_output = hf_dataset_path + "__bak_before_ase_fields_update"
    if os.path.exists(backup_output):
        shutil.rmtree(backup_output)
    os.replace(hf_dataset_path, backup_output)
    os.replace(temp_output, hf_dataset_path)
    shutil.rmtree(backup_output)

    return total_rows, missing_ase_matches, source_added, lowdin_charges_added, source_skipped


def main() -> None:
    
    hf_dataset_path = "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/huggingface_datasets/ani2x/ani2x_metadata"
    ase_db_path = "/Volumes/JAC_Backup/OpenFF/Meta-OMol25/ase_databases/ani2x.db"
    
    if not os.path.exists(hf_dataset_path):
        raise FileNotFoundError(f"HF dataset path not found: {hf_dataset_path}")
    if not os.path.exists(ase_db_path):
        raise FileNotFoundError(f"ASE DB path not found: {ase_db_path}")

    ase_index_map, ase_total, ase_lowdin_charges_rows = build_ase_index_map(ase_db_path, True)
    hf_total, missing, source_added, lowdin_charges_added, source_skipped = add_ase_fields_to_dataset(
        hf_dataset_path=hf_dataset_path,
        ase_index_map=ase_index_map,
        show_progress=True,
    )

    print(f"ASE rows indexed: {ase_total}")
    print(f"ASE rows with lowdin_charges: {ase_lowdin_charges_rows}")
    print(f"HF rows processed: {hf_total}")
    print(f"Rows without ASE match: {missing}")
    if source_skipped:
        print("source column already present in HF dataset; skipping source update")
    elif source_added:
        print("source column added to HF dataset")

    if lowdin_charges_added:
        print("lowdin_charges column added to HF dataset")
    elif ase_lowdin_charges_rows == 0:
        print("No lowdin_charges values found in ASE data; skipping lowdin_charges column")
    else:
        print("lowdin_charges column already present in HF dataset; skipping lowdin_charges update")

    print(f"Dataset update completed at: {hf_dataset_path}")


if __name__ == "__main__":
    main()
