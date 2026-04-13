import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENEMMA_ROOT = REPO_ROOT / "OpenEMMA"

for path in (REPO_ROOT, OPENEMMA_ROOT, REPO_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.main_v3 import extract_qwen_image_embeds
from training.train_scene_branch import (
    DEFAULT_DATAROOT,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_VERSION,
    NuScenesSceneBranchDataset,
    load_frozen_qwen,
    resolve_qwen_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen Qwen image embeddings for nuScenes CAM_FRONT samples."
    )
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    parser.add_argument("--qwen-model-path", type=str, default=None)
    parser.add_argument("--embeddings-dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    dataset = NuScenesSceneBranchDataset(
        dataroot=args.dataroot,
        version=args.version,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No CAM_FRONT key-frame samples were found in the requested nuScenes split.")

    print(f"Loaded {len(dataset)} samples from nuScenes {args.version}.")
    print("Using frozen Qwen device: cuda")
    qwen_source = resolve_qwen_source(args.qwen_model_path)
    print(f"Loading frozen Qwen feature extractor from: {qwen_source}")
    processor, qwen_model = load_frozen_qwen(qwen_source, torch.device("cuda"))

    for sample_index in range(len(dataset)):
        batch = dataset[sample_index]
        sample_token = batch["sample_token"]
        image_path = batch["image_path"]

        with torch.no_grad():
            image_embeds = extract_qwen_image_embeds(image_path, processor, qwen_model)

        emb_path = embeddings_dir / f"{sample_token}.pt"
        torch.save(image_embeds.cpu(), emb_path)

        if (sample_index + 1) % 20 == 0:
            print(
                f"Saved embeddings for sample {sample_index + 1}/{len(dataset)} | "
                f"token={sample_token}"
            )


if __name__ == "__main__":
    main()
