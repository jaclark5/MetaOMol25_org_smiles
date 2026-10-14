import argparse
import os

from datasets import load_from_disk
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def _config_exists_on_hub(api: HfApi, repo_id: str, config_name: str) -> bool:
    """Return True if a config directory already exists in a Hub dataset repo."""
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        # Repo does not exist yet, so the config cannot exist either.
        return False

    prefix = f"{config_name}/"
    return any(path == config_name or path.startswith(prefix) for path in repo_files)


def _wipe_config_on_hub(api: HfApi, repo_id: str, config_name: str) -> int:
    """Delete all files under a config directory in a Hub dataset repo."""
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return 0

    prefix = f"{config_name}/"
    files_to_delete = [path for path in repo_files if path.startswith(prefix)]
    for path in files_to_delete:
        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Delete existing files for config '{config_name}'",
            token="your_token",
        )
    return len(files_to_delete)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Push local Meta-OMol25 Hugging Face datasets to the Hub as two configs "
            "(metadata + smee)."
        )
    )
    parser.add_argument("--repo-id", required=True, help="Hub dataset repo, e.g. your-username/your-dataset")
    parser.add_argument("--metadata-path", required=True, help="Local metadata dataset directory")
    parser.add_argument("--smee-path", required=True, help="Local smee dataset directory")
    parser.add_argument("--metadata-config", default="metadata", help="Hub config name for metadata dataset")
    parser.add_argument("--smee-config", default="smee", help="Hub config name for smee dataset")
    parser.add_argument(
        "--if-config-exists",
        choices=["skip", "overwrite"],
        default="skip",
        help=(
            "Behavior when a config directory already exists on the Hub. "
            "Default: skip"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata dataset not found: {args.metadata_path}")
    if not os.path.exists(args.smee_path):
        raise FileNotFoundError(f"SMEE dataset not found: {args.smee_path}")

    print(f"Loading metadata dataset from: {args.metadata_path}")
    metadata_ds = load_from_disk(args.metadata_path)
    print(f"Loading smee dataset from: {args.smee_path}")
    smee_ds = load_from_disk(args.smee_path)
    api = HfApi()

    metadata_exists = _config_exists_on_hub(api, args.repo_id, args.metadata_config)
    if metadata_exists and args.if_config_exists == "skip":
        print(
            f"Skipping metadata push: config directory '{args.metadata_config}' "
            f"already exists in {args.repo_id}."
        )
    else:
        action = "Overwriting" if metadata_exists else "Pushing"
        if metadata_exists and args.if_config_exists == "overwrite":
            deleted_count = _wipe_config_on_hub(api, args.repo_id, args.metadata_config)
            print(
                f"Deleted {deleted_count} existing file(s) from "
                f"{args.repo_id}/{args.metadata_config} before re-upload."
            )
        print(
            f"{action} metadata to {args.repo_id} (config={args.metadata_config})"
        )
        metadata_ds.push_to_hub(
            args.repo_id,
            config_name=args.metadata_config,
            token="your_token",
        )

    smee_exists = _config_exists_on_hub(api, args.repo_id, args.smee_config)
    if smee_exists and args.if_config_exists == "skip":
        print(
            f"Skipping smee push: config directory '{args.smee_config}' "
            f"already exists in {args.repo_id}."
        )
    else:
        action = "Overwriting" if smee_exists else "Pushing"
        if smee_exists and args.if_config_exists == "overwrite":
            deleted_count = _wipe_config_on_hub(api, args.repo_id, args.smee_config)
            print(
                f"Deleted {deleted_count} existing file(s) from "
                f"{args.repo_id}/{args.smee_config} before re-upload."
            )
        print(
            f"{action} smee to {args.repo_id} (config={args.smee_config})"
        )
        smee_ds.push_to_hub(
            args.repo_id,
            config_name=args.smee_config,
            token="your_token",
        )

    print("Done: both configs were pushed to the Hub.")


if __name__ == "__main__":
    main()
