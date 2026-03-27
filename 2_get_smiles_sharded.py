import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import time
from typing import Dict, List

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
        "--report-every",
        type=int,
        default=10000,
        help="Print overall progress every N structures (default: 10000).",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    ds_name = args.ds_name
    shard_size = args.shard_size
    workers = max(1, args.workers)
    report_every = max(1, args.report_every)

    # Suppress per-shard HF dataset save bars so global progress lines stay readable.
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass

    ase_db = connect(os.path.join(dataset_path, f"ase_databases/{ds_name}.db"))
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
    print(f"Using {workers} conversion worker(s)")

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
        payloads: List[Dict] = []
        shard_read_s = 0.0
        shard_convert_s = 0.0
        shard_write_s = 0.0
        shard_failed_log_s = 0.0
        shard_checkpoint_s = 0.0
        for local_i, row in enumerate(ase_db.select(offset=shard_start, limit=(shard_stop - shard_start))):
            i = shard_start + local_i
            t_read = time.perf_counter()
            atoms = row.toatoms(add_additional_information=True)
            atoms.info.update(row.data)
            shard_read_s += time.perf_counter() - t_read

            key_matches = [x for x in atoms.info.keys() if "unique" in x]
            key = key_matches[0] if key_matches else "data_id"
            payloads.append(
                {
                    "index": i,
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
            )

        t_convert = time.perf_counter()
        if workers == 1:
            results_iter = map(process_payload, payloads)
            for result in results_iter:
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
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results_iter = executor.map(process_payload, payloads)
                for result in results_iter:
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

        shard_convert_s += time.perf_counter() - t_convert
        del payloads

        if records:
            t_write = time.perf_counter()
            metadata_records = [
                {
                    k: record[k]
                    for k in metadata_cols
                }
                for record in records
            ]
            metadata_shard = Dataset.from_list(metadata_records)
            metadata_shard.save_to_disk(metadata_shard_path)

            smee_entries = [
                {
                    "smiles": record["smiles"],
                    "coords": record["coords"],
                    "energy": record["energy"],
                    "forces": record["forces"],
                }
                for record in records
            ]
            smee_shard = descent.targets.energy.create_dataset(smee_entries)
            smee_shard.save_to_disk(smee_shard_path)

            del smee_shard
            del smee_entries
            del metadata_shard
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

    print(f"Sharded outputs written under: {output_path}")
    print(f"Failed rows log: {failed_rows_path}")

    metadata_shard_names = sorted(
        name
        for name in os.listdir(metadata_shards_dir)
        if os.path.isdir(os.path.join(metadata_shards_dir, name))
    )
    smee_shard_names = sorted(
        name
        for name in os.listdir(smee_shards_dir)
        if os.path.isdir(os.path.join(smee_shards_dir, name))
    )

    if os.path.exists(metadata_final_path):
        print(f"Metadata dataset already exists at {metadata_final_path}; skipping merge.")
    elif metadata_shard_names:
        print(f"Streaming-merge metadata shards into {metadata_final_path}")
        metadata_dataset = Dataset.from_generator(
            shard_row_generator,
            gen_kwargs={
                "shards_dir": metadata_shards_dir,
                "shard_names": metadata_shard_names,
            },
        )
        metadata_dataset.save_to_disk(metadata_final_path)
        print(f"Merged metadata dataset written to {metadata_final_path}")
    else:
        print("No metadata shards found to merge.")

    if os.path.exists(smee_final_path):
        print(f"Smee dataset already exists at {smee_final_path}; skipping merge.")
    elif smee_shard_names:
        print(f"Streaming-merge smee shards into {smee_final_path}")
        smee_dataset = Dataset.from_generator(
            shard_row_generator,
            gen_kwargs={
                "shards_dir": smee_shards_dir,
                "shard_names": smee_shard_names,
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
