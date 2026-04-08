import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import gc
import json
import os
import shutil
import time
import re
from typing import Dict, List
import warnings

from ase.db import connect
import numpy as np
from scipy.constants import Avogadro, calorie, physical_constants
from datasets import Dataset, load_from_disk
from openff.toolkit import Molecule
import descent.targets.energy

# ASE returns energies in eV and forces in eV/A.
# Convert to requested units: kcal/mol and kcal/mol/A.
EV_JOULE = physical_constants["electron volt-joule relationship"][0]
EV_TO_KCAL_PER_MOL: float = EV_JOULE * Avogadro / (1000.0 * calorie)
SHARD_NAME_RE = re.compile(r"^shard_(\d{9})_(\d{9})$")


def configure_runtime_warnings(suppress_toolkit_warnings: bool) -> None:
    """Reduce warning/log spam that can dominate shared filesystem I/O."""
    if not suppress_toolkit_warnings:
        return

    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    try:
        from openbabel import openbabel as ob

        ob.obErrorLog.SetOutputLevel(0)
    except Exception:
        pass

def check_elements(atoms, metal_elements, ligand_elements, max_num_metals, min_num_lig_el=1):
    symbols = atoms.get_chemical_symbols()
    check_all = all(el in metal_elements or el in ligand_elements for el in symbols)
    if not check_all:
        return False, False

    if max_num_metals > 0:
        nmetals = np.sum([el in metal_elements for el in symbols])
        check_nmetals = nmetals <= max_num_metals and nmetals > 0
    else:
        check_nmetals = True

    nligandelements = np.sum([el in ligand_elements for el in symbols])
    check_ligands = nligandelements >= min_num_lig_el
    return check_ligands, check_nmetals


def extract_data(atoms, index, method="custom"):
    import tmos

    rdmol = tmos.build_rdmol.xyz_to_rdkit(
        atoms.get_chemical_symbols(),
        atoms.positions,
        ignore_scale=True,
        method=method,
    )
    rdmol = tmos.build_rdmol.determine_bonds(
        rdmol,
        charge=atoms.info["charge"],
        custom_cleanup=True if method == "custom" else False,
    )
    smiles = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True).to_smiles(
        mapped=True,
        isomeric=True,
        explicit_hydrogens=True,
    )

    dict_metadata = {
        "OMol25_data_id": atoms.info["data_id"],
        "OMol25_id": atoms.info["omol25-index"],
        "OMol25_split": atoms.info["omol25-split"],
        "OpenFF_id": index,
        "formula": atoms.get_chemical_formula(),
        "charge": atoms.info["charge"],
        "OpenFF_Elements": check_elements(atoms, [], ["C", "H", "P", "S", "O", "N", "F", "Cl", "Br", "I"], 0)[0],
        "OpenFF_abs(q)<=1": abs(atoms.info["charge"]) <= 1,
        "OpenFF_spin=1": atoms.info["spin"] == 1,
        "smiles": smiles,
    }

    dict_smee = {
        "smiles": smiles,
        "coords": atoms.positions.tolist(),
        "energy": float(atoms.get_total_energy()) * EV_TO_KCAL_PER_MOL,
        "forces": (atoms.get_forces() * EV_TO_KCAL_PER_MOL).tolist(),
    }

    return dict_metadata, dict_smee


def extract_data_from_payload(payload: Dict, method: str = "custom"):
    import tmos

    symbols = payload["symbols"]
    positions = payload["positions"]
    charge = payload["charge"]

    rdmol = tmos.build_rdmol.xyz_to_rdkit(
        symbols,
        positions,
        ignore_scale=True,
        method=method,
    )
    rdmol = tmos.build_rdmol.determine_bonds(
        rdmol,
        charge=charge,
        custom_cleanup=True if method == "custom" else False,
    )
    smiles = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True).to_smiles(
        mapped=True,
        isomeric=True,
        explicit_hydrogens=True,
    )

    dict_metadata = {
        "OMol25_data_id": payload["data_id"],
        "OMol25_id": payload["omol25_index"],
        "OMol25_split": payload["omol25_split"],
        "OpenFF_id": payload["index"],
        "formula": payload["formula"],
        "charge": charge,
        "OpenFF_Elements": all(el in ["C", "H", "P", "S", "O", "N", "F", "Cl", "Br", "I"] for el in symbols),
        "OpenFF_abs(q)<=1": abs(charge) <= 1,
        "OpenFF_spin=1": payload["spin"] == 1,
        "smiles": smiles,
    }

    dict_smee = {
        "smiles": smiles,
        "coords": positions,
        "energy": payload["energy_ev"] * EV_TO_KCAL_PER_MOL,
        "forces": [[f * EV_TO_KCAL_PER_MOL for f in xyz] for xyz in payload["forces_ev"]],
    }

    return dict_metadata, dict_smee


def process_payload(payload: Dict) -> Dict:
    try:
        dict_metadata, dict_smee = extract_data_from_payload(payload)
        return {"ok": True, "record": {**dict_metadata, **dict_smee}}
    except Exception as e:
        try:
            dict_metadata, dict_smee = extract_data_from_payload(payload, method="openbabel")
            return {"ok": True, "record": {**dict_metadata, **dict_smee}}
        except Exception:
            return {
                "ok": False,
                "index": payload["index"],
                "ase_id": payload["ase_id"],
                "error": repr(e),
            }


def maybe_report(done: int, total: int, failed: int, state: Dict) -> None:
    if total == 0:
        return
    while done >= state["next_done"] and state["next_done"] <= total:
        mark = state["next_done"]
        pct = 100.0 * mark / total
        print(f"Processed {mark}/{total} ({pct:.2f}%) | failed={failed}")
        state["next_done"] += state["step"]
    if done == total and not state["done_printed"]:
        pct = 100.0
        print(f"Processed {done}/{total} ({pct:.2f}%) | failed={failed}")
        state["done_printed"] = True


def write_checkpoint(checkpoint_path: str, next_offset: int, failed_count: int) -> None:
    with open(checkpoint_path, "w") as f:
        json.dump({"next_offset": next_offset, "failed_count": failed_count}, f, indent=2)


def shard_row_generator(shards_dir: str, shard_names: List[str]):
    for name in shard_names:
        shard_path = os.path.join(shards_dir, name)
        shard_ds = load_from_disk(shard_path)
        for i in range(len(shard_ds)):
            yield shard_ds[i]


def parse_shard_name(name: str):
    match = SHARD_NAME_RE.match(name)
    if not match:
        return None
    start = int(match.group(1))
    stop = int(match.group(2))
    if stop < start:
        return None
    return start, stop


def sorted_shard_names(names: List[str]) -> List[str]:
    def key_func(name: str):
        parsed = parse_shard_name(name)
        if parsed is None:
            return (10**18, name)
        return (parsed[0], name)

    return sorted(names, key=key_func)


def discover_valid_shards(shards_dir: str, label: str):
    valid_names: List[str] = []
    invalid_names: List[str] = []

    for name in sorted_shard_names(os.listdir(shards_dir)):
        shard_path = os.path.join(shards_dir, name)
        if not os.path.isdir(shard_path):
            continue

        parsed = parse_shard_name(name)
        if parsed is None:
            invalid_names.append(name)
            continue

        try:
            load_from_disk(shard_path)
            valid_names.append(name)
        except Exception as e:
            print(f"Skipping invalid {label} shard {name}: {e}")
            invalid_names.append(name)

    return valid_names, invalid_names


def collect_index_coverage(
    total_rows: int,
    metadata_shards_dir: str,
    metadata_shard_names: List[str],
    failed_rows_path: str,
) -> Dict:
    success_seen = bytearray(total_rows)
    failed_seen = bytearray(total_rows)

    success_unique = 0
    success_duplicates = 0
    success_out_of_range = 0

    for name in metadata_shard_names:
        shard_path = os.path.join(metadata_shards_dir, name)
        shard_ds = load_from_disk(shard_path)
        if "OpenFF_id" not in shard_ds.column_names:
            raise RuntimeError(f"Shard {name} is missing required OpenFF_id column")

        for idx in shard_ds["OpenFF_id"]:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= total_rows:
                success_out_of_range += 1
                continue
            if success_seen[idx_int]:
                success_duplicates += 1
            else:
                success_seen[idx_int] = 1
                success_unique += 1

    failed_unique = 0
    failed_duplicates = 0
    failed_out_of_range = 0
    failed_parse_errors = 0

    if os.path.exists(failed_rows_path):
        with open(failed_rows_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    idx_int = int(payload["index"])
                except Exception:
                    failed_parse_errors += 1
                    if failed_parse_errors <= 5:
                        print(f"Could not parse failed row entry at line {line_no}: {text[:200]}")
                    continue

                if idx_int < 0 or idx_int >= total_rows:
                    failed_out_of_range += 1
                    continue
                if failed_seen[idx_int]:
                    failed_duplicates += 1
                else:
                    failed_seen[idx_int] = 1
                    failed_unique += 1

    overlap_count = 0
    missing_count = 0
    missing_examples: List[int] = []
    missing_indices: List[int] = []
    for idx in range(total_rows):
        in_success = success_seen[idx] == 1
        in_failed = failed_seen[idx] == 1
        if in_success and in_failed:
            overlap_count += 1
        elif (not in_success) and (not in_failed):
            missing_count += 1
            missing_indices.append(idx)
            if len(missing_examples) < 20:
                missing_examples.append(idx)

    expected_accounted = total_rows
    accounted_unique = success_unique + failed_unique - overlap_count

    return {
        "success_unique": success_unique,
        "failed_unique": failed_unique,
        "overlap_count": overlap_count,
        "accounted_unique": accounted_unique,
        "expected_accounted": expected_accounted,
        "missing_count": missing_count,
        "missing_examples": missing_examples,
        "missing_indices": missing_indices,
        "success_duplicates": success_duplicates,
        "failed_duplicates": failed_duplicates,
        "success_out_of_range": success_out_of_range,
        "failed_out_of_range": failed_out_of_range,
        "failed_parse_errors": failed_parse_errors,
    }


def format_index_ranges(indices: List[int], max_ranges: int = 10) -> str:
    if not indices:
        return "[]"

    ranges = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))

    parts = []
    for a, b in ranges[:max_ranges]:
        if a == b:
            parts.append(str(a))
        else:
            parts.append(f"{a}-{b}")
    if len(ranges) > max_ranges:
        parts.append("...")
    return "[" + ", ".join(parts) + "]"


def validate_index_coverage(
    total_rows: int,
    metadata_shards_dir: str,
    metadata_shard_names: List[str],
    failed_rows_path: str,
) -> None:
    coverage = collect_index_coverage(
        total_rows=total_rows,
        metadata_shards_dir=metadata_shards_dir,
        metadata_shard_names=metadata_shard_names,
        failed_rows_path=failed_rows_path,
    )

    print(
        "Coverage summary: "
        f"success_unique={coverage['success_unique']}, "
        f"failed_unique={coverage['failed_unique']}, "
        f"overlap={coverage['overlap_count']}, "
        f"accounted_unique={coverage['accounted_unique']}/{coverage['expected_accounted']}, "
        f"missing={coverage['missing_count']}"
    )

    problems = []
    if coverage["missing_count"] > 0:
        problems.append(
            "missing "
            f"{coverage['missing_count']} indices "
            f"(examples: {coverage['missing_examples']}; "
            f"ranges: {format_index_ranges(coverage['missing_indices'])})"
        )
    if coverage["success_duplicates"] > 0:
        problems.append(f"{coverage['success_duplicates']} duplicate OpenFF_id entries across metadata shards")
    if coverage["failed_duplicates"] > 0:
        problems.append(f"{coverage['failed_duplicates']} duplicate indices in failed_rows jsonl")
    if coverage["success_out_of_range"] > 0:
        problems.append(f"{coverage['success_out_of_range']} out-of-range OpenFF_id values in metadata shards")
    if coverage["failed_out_of_range"] > 0:
        problems.append(f"{coverage['failed_out_of_range']} out-of-range indices in failed_rows jsonl")
    if coverage["failed_parse_errors"] > 0:
        problems.append(f"{coverage['failed_parse_errors']} malformed lines in failed_rows jsonl")

    if coverage["overlap_count"] > 0:
        print(
            "Warning: "
            f"{coverage['overlap_count']} indices are present in both successful shards and failed_rows."
        )

    if problems:
        raise RuntimeError(
            "Coverage validation failed before merge: " + "; ".join(problems)
        )


def payload_from_row(row, index: int) -> Dict:
    atoms = row.toatoms(add_additional_information=True)
    atoms.info.update(row.data)

    key_matches = [x for x in atoms.info.keys() if "unique" in x]
    key = key_matches[0] if key_matches else "data_id"

    return {
        "index": index,
        "ase_id": atoms.info.get(key),
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.positions.tolist(),
        "formula": atoms.get_chemical_formula(),
        "charge": atoms.info["charge"],
        "spin": atoms.info["spin"],
        "data_id": atoms.info["data_id"],
        "omol25_index": atoms.info["omol25-index"],
        "omol25_split": atoms.info["omol25-split"],
        "energy_ev": float(atoms.get_total_energy()),
        "forces_ev": atoms.get_forces().tolist(),
    }


def save_metadata_shard(metadata_records: List[Dict], shard_path: str) -> None:
    metadata_shard = Dataset.from_list(metadata_records)
    tmp_path = f"{shard_path}.tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    metadata_shard.save_to_disk(tmp_path)
    if os.path.exists(shard_path):
        shutil.rmtree(shard_path)
    os.replace(tmp_path, shard_path)


def save_smee_shard(smee_entries: List[Dict], shard_path: str) -> None:
    smee_shard = descent.targets.energy.create_dataset(smee_entries)
    tmp_path = f"{shard_path}.tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    smee_shard.save_to_disk(tmp_path)
    if os.path.exists(shard_path):
        shutil.rmtree(shard_path)
    os.replace(tmp_path, shard_path)


def rerun_missing_indices(
    ase_db,
    missing_indices: List[int],
    metadata_cols: List[str],
    metadata_shards_dir: str,
    smee_shards_dir: str,
    failed_rows_path: str,
    workers: int,
    executor_kind: str,
    max_inflight: int,
    write_workers: int,
    suppress_toolkit_warnings: bool,
) -> int:
    if not missing_indices:
        return 0

    missing_sorted = sorted(set(int(i) for i in missing_indices))
    print(
        f"Rerunning {len(missing_sorted)} missing index/indices "
        f"across ranges {format_index_ranges(missing_sorted)}"
    )

    ranges = []
    start = missing_sorted[0]
    prev = missing_sorted[0]
    for idx in missing_sorted[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))

    failed_count = 0
    executor = None
    try:
        if workers > 1:
            if executor_kind == "process":
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=configure_runtime_warnings,
                    initargs=(suppress_toolkit_warnings,),
                )
            else:
                executor = ThreadPoolExecutor(max_workers=workers)

        for range_start, range_stop in ranges:
            shard_name = f"shard_{range_start:09d}_{range_stop:09d}"
            metadata_shard_path = os.path.join(metadata_shards_dir, shard_name)
            smee_shard_path = os.path.join(smee_shards_dir, shard_name)

            records: List[Dict] = []
            failed_rows_batch: List[Dict] = []

            def consume_result(result: Dict) -> None:
                nonlocal failed_count
                if result["ok"]:
                    records.append(result["record"])
                else:
                    failed_count += 1
                    failed_rows_batch.append(
                        {
                            "index": result["index"],
                            "ase_id": result["ase_id"],
                            "error": result["error"],
                        }
                    )

            count = range_stop - range_start + 1
            if workers == 1:
                for local_i, row in enumerate(ase_db.select(offset=range_start, limit=count)):
                    i = range_start + local_i
                    payload = payload_from_row(row, i)
                    consume_result(process_payload(payload))
            else:
                pending = set()
                for local_i, row in enumerate(ase_db.select(offset=range_start, limit=count)):
                    i = range_start + local_i
                    payload = payload_from_row(row, i)
                    pending.add(executor.submit(process_payload, payload))
                    if len(pending) >= max_inflight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            consume_result(future.result())

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume_result(future.result())

            if records:
                metadata_records = [
                    {
                        k: record[k]
                        for k in metadata_cols
                    }
                    for record in records
                ]
                smee_entries = [
                    {
                        "smiles": record["smiles"],
                        "coords": record["coords"],
                        "energy": record["energy"],
                        "forces": record["forces"],
                    }
                    for record in records
                ]

                if write_workers > 1:
                    with ThreadPoolExecutor(max_workers=2) as write_executor:
                        fut_meta = write_executor.submit(save_metadata_shard, metadata_records, metadata_shard_path)
                        fut_smee = write_executor.submit(save_smee_shard, smee_entries, smee_shard_path)
                        fut_meta.result()
                        fut_smee.result()
                else:
                    save_metadata_shard(metadata_records, metadata_shard_path)
                    save_smee_shard(smee_entries, smee_shard_path)

            if failed_rows_batch:
                with open(failed_rows_path, "a") as f:
                    for failed_row in failed_rows_batch:
                        f.write(json.dumps(failed_row) + "\n")

            print(
                f"Rerun shard {shard_name}: success={len(records)}, failed={len(failed_rows_batch)}"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return failed_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenFF shards with checkpointed processing.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-path")
    parser.add_argument("--ds-name")
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
        help="Parallel conversion workers for per-structure chemistry conversion.",
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Parallel executor backend for conversion workers.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="Max in-flight conversion tasks; 0 uses 4x workers.",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=1,
        help="Number of threads for per-shard writes (1 or 2).",
    )
    parser.add_argument(
        "--suppress-toolkit-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress verbose toolkit warning logs that can bottleneck I/O.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10000,
        help="Print overall progress every N structures (default: 10000).",
    )
    parser.add_argument(
        "--repair-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before merge, rerun indices missing from both shards and failed_rows.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    ds_name = args.ds_name
    shard_size = args.shard_size
    workers = max(1, args.workers)
    executor_kind = args.executor
    max_inflight = args.max_inflight if args.max_inflight > 0 else max(4, workers * 4)
    write_workers = max(1, min(2, args.write_workers))
    report_every = max(1, args.report_every)

    configure_runtime_warnings(args.suppress_toolkit_warnings)

    # Suppress per-shard HF dataset save bars so global progress lines stay readable.
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass

    ase_db = connect(os.path.join(dataset_path, f"{ds_name}.db"))
    output_path = os.path.join(output_path, "huggingface_datasets")
    os.makedirs(output_path, exist_ok=True)

    metadata_cols = [
        "OMol25_data_id",
        "OMol25_id",
        "OMol25_split",
        "OpenFF_id",
        "formula",
        "charge",
        "OpenFF_Elements",
        "OpenFF_abs(q)<=1",
        "OpenFF_spin=1",
        "smiles",
    ]

    total_rows = ase_db.count()
    print(f"Total rows to process: {total_rows}")
    print(
        f"Using {workers} conversion worker(s) "
        f"(executor={executor_kind}, max_inflight={max_inflight}, write_workers={write_workers})"
    )

    metadata_shards_dir = os.path.join(output_path, f"{ds_name}_metadata_shards")
    smee_shards_dir = os.path.join(output_path, f"{ds_name}_smee_shards")
    metadata_final_path = os.path.join(output_path, f"{ds_name}_metadata")
    smee_final_path = os.path.join(output_path, f"{ds_name}_smee")
    checkpoint_path = os.path.join(output_path, f"{ds_name}_checkpoint.json")
    failed_rows_path = os.path.join(output_path, f"{ds_name}_failed_rows.jsonl")

    for p in [metadata_shards_dir, smee_shards_dir]:
        os.makedirs(p, exist_ok=True)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        next_offset = int(checkpoint.get("next_offset", 0))
        failed_count = int(checkpoint.get("failed_count", 0))
        print(f"Resuming from offset {next_offset} (failed so far: {failed_count})")
    else:
        next_offset = 0
        failed_count = 0
        write_checkpoint(checkpoint_path, next_offset, failed_count)

    report_state = {
        "next_done": ((next_offset // report_every) + 1) * report_every,
        "step": report_every,
        "done_printed": False,
    }
    maybe_report(next_offset, total_rows, failed_count, report_state)

    perf_totals = {
        "read_s": 0.0,
        "convert_s": 0.0,
        "write_s": 0.0,
        "failed_log_s": 0.0,
        "checkpoint_s": 0.0,
    }

    executor = None
    try:
        if workers > 1:
            if executor_kind == "process":
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=configure_runtime_warnings,
                    initargs=(args.suppress_toolkit_warnings,),
                )
            else:
                executor = ThreadPoolExecutor(max_workers=workers)

        while next_offset < total_rows:
            shard_t0 = time.perf_counter()
            shard_start = next_offset
            shard_stop = min(next_offset + shard_size, total_rows)
            shard_name = f"shard_{shard_start:09d}_{shard_stop - 1:09d}"

            metadata_shard_path = os.path.join(metadata_shards_dir, shard_name)
            smee_shard_path = os.path.join(smee_shards_dir, shard_name)
            required_paths = [metadata_shard_path, smee_shard_path]

            if all(os.path.exists(p) for p in required_paths):
                next_offset = shard_stop
                t_ckpt = time.perf_counter()
                write_checkpoint(checkpoint_path, next_offset, failed_count)
                perf_totals["checkpoint_s"] += time.perf_counter() - t_ckpt
                maybe_report(next_offset, total_rows, failed_count, report_state)
                continue

            records: List[Dict] = []
            failed_rows_batch: List[Dict] = []
            shard_read_s = 0.0
            shard_convert_s = 0.0
            shard_write_s = 0.0
            shard_failed_log_s = 0.0
            shard_checkpoint_s = 0.0

            def consume_result(result: Dict) -> None:
                nonlocal failed_count
                if result["ok"]:
                    records.append(result["record"])
                else:
                    failed_count += 1
                    failed_rows_batch.append(
                        {
                            "index": result["index"],
                            "ase_id": result["ase_id"],
                            "error": result["error"],
                        }
                    )

            t_convert = time.perf_counter()
            if workers == 1:
                for local_i, row in enumerate(ase_db.select(offset=shard_start, limit=(shard_stop - shard_start))):
                    i = shard_start + local_i
                    t_read = time.perf_counter()
                    payload = payload_from_row(row, i)
                    shard_read_s += time.perf_counter() - t_read
                    consume_result(process_payload(payload))
            else:
                pending = set()
                for local_i, row in enumerate(ase_db.select(offset=shard_start, limit=(shard_stop - shard_start))):
                    i = shard_start + local_i
                    t_read = time.perf_counter()
                    payload = payload_from_row(row, i)
                    shard_read_s += time.perf_counter() - t_read
                    pending.add(executor.submit(process_payload, payload))

                    if len(pending) >= max_inflight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            consume_result(future.result())

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume_result(future.result())

            shard_convert_s += time.perf_counter() - t_convert

            if records:
                t_write = time.perf_counter()
                metadata_records = [
                    {
                        k: record[k]
                        for k in metadata_cols
                    }
                    for record in records
                ]
                smee_entries = [
                    {
                        "smiles": record["smiles"],
                        "coords": record["coords"],
                        "energy": record["energy"],
                        "forces": record["forces"],
                    }
                    for record in records
                ]

                if write_workers > 1:
                    with ThreadPoolExecutor(max_workers=2) as write_executor:
                        fut_meta = write_executor.submit(save_metadata_shard, metadata_records, metadata_shard_path)
                        fut_smee = write_executor.submit(save_smee_shard, smee_entries, smee_shard_path)
                        fut_meta.result()
                        fut_smee.result()
                else:
                    save_metadata_shard(metadata_records, metadata_shard_path)
                    save_smee_shard(smee_entries, smee_shard_path)

                del smee_entries
                del metadata_records
                shard_write_s += time.perf_counter() - t_write

            if failed_rows_batch:
                t_failed_log = time.perf_counter()
                with open(failed_rows_path, "a") as f:
                    for failed_row in failed_rows_batch:
                        f.write(json.dumps(failed_row) + "\n")
                shard_failed_log_s += time.perf_counter() - t_failed_log

            del records
            del failed_rows_batch
            gc.collect()

            next_offset = shard_stop
            t_ckpt = time.perf_counter()
            write_checkpoint(checkpoint_path, next_offset, failed_count)
            shard_checkpoint_s += time.perf_counter() - t_ckpt
            maybe_report(next_offset, total_rows, failed_count, report_state)

            perf_totals["read_s"] += shard_read_s
            perf_totals["convert_s"] += shard_convert_s
            perf_totals["write_s"] += shard_write_s
            perf_totals["failed_log_s"] += shard_failed_log_s
            perf_totals["checkpoint_s"] += shard_checkpoint_s

            shard_elapsed_s = time.perf_counter() - shard_t0
            print(
                f"Shard {shard_name}: elapsed={shard_elapsed_s:.1f}s "
                f"read={shard_read_s:.1f}s convert={shard_convert_s:.1f}s "
                f"write={shard_write_s:.1f}s"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"Sharded outputs written under: {output_path}")
    print(f"Failed rows log: {failed_rows_path}")

    metadata_shard_names, metadata_invalid = discover_valid_shards(metadata_shards_dir, "metadata")
    smee_shard_names, smee_invalid = discover_valid_shards(smee_shards_dir, "smee")

    metadata_set = set(metadata_shard_names)
    smee_set = set(smee_shard_names)
    shared_shards = sorted_shard_names(list(metadata_set & smee_set))
    metadata_only = sorted_shard_names(list(metadata_set - smee_set))
    smee_only = sorted_shard_names(list(smee_set - metadata_set))

    if metadata_invalid:
        print(f"Ignored {len(metadata_invalid)} invalid metadata shard directories.")
    if smee_invalid:
        print(f"Ignored {len(smee_invalid)} invalid smee shard directories.")
    if metadata_only:
        print(
            "Warning: metadata-only shards will be excluded from merge "
            f"({len(metadata_only)} shard(s), first: {metadata_only[:5]})."
        )
    if smee_only:
        print(
            "Warning: smee-only shards will be excluded from merge "
            f"({len(smee_only)} shard(s), first: {smee_only[:5]})."
        )

    if shared_shards:
        if args.repair_missing:
            coverage = collect_index_coverage(
                total_rows=total_rows,
                metadata_shards_dir=metadata_shards_dir,
                metadata_shard_names=shared_shards,
                failed_rows_path=failed_rows_path,
            )
            if coverage["missing_count"] > 0:
                print(
                    "Detected missing indices before merge; attempting repair rerun "
                    f"for {coverage['missing_count']} entries."
                )
                rerun_failed = rerun_missing_indices(
                    ase_db=ase_db,
                    missing_indices=coverage["missing_indices"],
                    metadata_cols=metadata_cols,
                    metadata_shards_dir=metadata_shards_dir,
                    smee_shards_dir=smee_shards_dir,
                    failed_rows_path=failed_rows_path,
                    workers=workers,
                    executor_kind=executor_kind,
                    max_inflight=max_inflight,
                    write_workers=write_workers,
                    suppress_toolkit_warnings=args.suppress_toolkit_warnings,
                )
                print(f"Repair rerun complete (new failed rows during rerun: {rerun_failed}).")

                metadata_shard_names, metadata_invalid = discover_valid_shards(metadata_shards_dir, "metadata")
                smee_shard_names, smee_invalid = discover_valid_shards(smee_shards_dir, "smee")
                metadata_set = set(metadata_shard_names)
                smee_set = set(smee_shard_names)
                shared_shards = sorted_shard_names(list(metadata_set & smee_set))

        print("Validating index coverage across successful shards and failed rows...")
        validate_index_coverage(
            total_rows=total_rows,
            metadata_shards_dir=metadata_shards_dir,
            metadata_shard_names=shared_shards,
            failed_rows_path=failed_rows_path,
        )
        print("Coverage validation passed.")

    if os.path.exists(metadata_final_path):
        print(f"Metadata dataset already exists at {metadata_final_path}; skipping merge.")
    elif shared_shards:
        print(f"Streaming-merge metadata shards into {metadata_final_path}")
        metadata_dataset = Dataset.from_generator(
            shard_row_generator,
            gen_kwargs={
                "shards_dir": metadata_shards_dir,
                "shard_names": shared_shards,
            },
        )
        metadata_dataset.save_to_disk(metadata_final_path)
        print(f"Merged metadata dataset written to {metadata_final_path}")
    else:
        print("No metadata shards found to merge.")

    if os.path.exists(smee_final_path):
        print(f"Smee dataset already exists at {smee_final_path}; skipping merge.")
    elif shared_shards:
        print(f"Streaming-merge smee shards into {smee_final_path}")
        smee_dataset = Dataset.from_generator(
            shard_row_generator,
            gen_kwargs={
                "shards_dir": smee_shards_dir,
                "shard_names": shared_shards,
            },
        )
        smee_dataset.save_to_disk(smee_final_path)
        print(f"Merged smee dataset written to {smee_final_path}")
    else:
        print("No smee shards found to merge.")

    print(f"Smee shards directory: {smee_shards_dir}")
    print(
        "Timing totals: "
        f"read={perf_totals['read_s']:.1f}s "
        f"convert={perf_totals['convert_s']:.1f}s "
        f"write={perf_totals['write_s']:.1f}s "
        f"failed_log={perf_totals['failed_log_s']:.1f}s "
        f"checkpoint={perf_totals['checkpoint_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
