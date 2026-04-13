import argparse
import os

from datasets import load_from_disk


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
    args = parser.parse_args()

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata dataset not found: {args.metadata_path}")
    if not os.path.exists(args.smee_path):
        raise FileNotFoundError(f"SMEE dataset not found: {args.smee_path}")

    print(f"Loading metadata dataset from: {args.metadata_path}")
    metadata_ds = load_from_disk(args.metadata_path)
    print(f"Loading smee dataset from: {args.smee_path}")
    smee_ds = load_from_disk(args.smee_path)

    print(
        f"Pushing metadata to {args.repo_id} (config={args.metadata_config}) "
    )
    metadata_ds.push_to_hub(
        args.repo_id,
        config_name=args.metadata_config,
    )

    print(
        f"Pushing smee to {args.repo_id} (config={args.smee_config}) "
    )
    smee_ds.push_to_hub(
        args.repo_id,
        config_name=args.smee_config,
    )

    print("Done: both configs were pushed to the Hub.")


if __name__ == "__main__":
    main()
